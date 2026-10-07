"""Run a reduced-width CPU probe against local, unmodified Forge source files.

No network, no checkpoint downloads, no writes to the provided source files.
This uses actual Krea2 class/method definitions, but substitutes CPU backend
operators and random tiny weights. It is NOT a pretrained-model/GPU test.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from forge_neo_nag.adapters.krea2 import Krea2Adapter, nag_block
from forge_neo_nag.config import NAGConfig
from forge_neo_nag.host import NAGModelWrapper, HostBindings
from tests.helpers import OPS, EmbedND, Predictor
from tests.test_adapter import reference_block

EXPECTED = {
    "krea": "eb5d997633fe0018e1ed5275865c4b7809f9372f",
    "kmodel": "0f09636d1e9919ef54c1a47c6c034ab08a922cc4",
}


def load_definitions(path: Path, key: str):
    data = path.read_bytes()
    digest = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    # Normalize Windows checkout line endings for comparison only.
    normalized = data.replace(b"\r\n", b"\n")
    digest_lf = hashlib.sha1(f"blob {len(normalized)}\0".encode() + normalized).hexdigest()
    tree = ast.parse(data.decode("utf-8-sig"), filename=str(path))
    tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    ns = {
        "torch": torch, "nn": nn, "F": F, "Optional": Optional, "math": math,
        "rearrange": rearrange, "EmbedND": EmbedND,
        "dynamic_args": SimpleNamespace(ref_latents=[]),
        "attention_function": OPS.attention,
        "cast_to": lambda tensor, **kwargs: tensor.to(**kwargs),
        "timestep_embedding": OPS.timestep_embedding,
        "pad_to_patch_size": OPS.pad,
        "ck": SimpleNamespace(apply_rope=OPS.rope),
    }
    exec(compile(tree, str(path), "exec"), ns)
    return ns, {"raw_git_blob": digest, "lf_git_blob": digest_lf,
                "matches_reviewed_source": digest_lf == EXPECTED[key]}


@torch.inference_mode()
def run(krea_source: Path, kmodel_source: Path):
    torch.set_num_threads(1)
    torch.manual_seed(98331)
    definitions, krea_hash = load_definitions(krea_source, "krea")
    kdefs, kmodel_hash = load_definitions(kmodel_source, "kmodel")
    DiT, KModel = definitions["SingleStreamDiT"], kdefs["KModel"]
    results = []
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for batch in (1, 2):
            model = DiT(features=64,tdim=16,txtdim=16,heads=4,kvheads=2,multiplier=1,
                        layers=2,channels=4,txtlayers=3,txtheads=2,txtkvheads=2)
            for name, parameter in model.named_parameters():
                if name.endswith("scale"):
                    parameter.zero_()
                else:
                    parameter.normal_(0, .08)
            model = model.to(dtype).eval()
            x = torch.randn(batch,4,1,5,7).to(dtype)
            context = torch.randn(batch,7,3,16).to(dtype)
            negative = torch.randn(1,5,3,16).to(dtype)
            t = torch.full((batch,),.6)
            tolerance = 2e-6 if dtype == torch.float32 else .008
            relative = 2e-5 if dtype == torch.float32 else .02
            cfg_zero = NAGConfig.parse(True,"wings",alpha=0)
            native = model(x.clone(),t,context.clone())
            adapted = Krea2Adapter(model,negative,cfg_zero,OPS)(x,t,context)
            torch.testing.assert_close(adapted,native,rtol=relative,atol=tolerance)
            results.append({"case":"native_forward_zero_alpha","dtype":str(dtype),"batch":batch,
                            "max_abs_error":float((adapted-native).abs().max()),"pass":True})

            cfg = NAGConfig.parse(True,"wings")
            pl,nl,il = 7,5,6
            pt,nt,im = torch.randn(batch,pl,64).to(dtype),torch.randn(batch,nl,64).to(dtype),torch.randn(batch,il,64).to(dtype)
            vec = (torch.randn(batch,1,384)*.1).to(dtype)
            image_ids=torch.zeros(batch,il,3)
            image_ids[:,:,1]=torch.arange(il)
            pf=model.pe_embedder(torch.cat((torch.zeros(batch,pl,3),image_ids),1))
            nf=model.pe_embedder(torch.cat((torch.zeros(batch,nl,3),image_ids),1))
            ntf=model.pe_embedder(torch.zeros(batch,nl,3))
            reference = reference_block(model.blocks[0],pt,nt,im,vec,pf,nf,cfg,{})
            optimized = nag_block(model.blocks[0],pt,nt,im,vec,pf,ntf,cfg,{},OPS)
            for actual,expected in zip(optimized,reference):
                torch.testing.assert_close(actual,expected,rtol=relative,atol=tolerance)
            results.append({"case":"native_block_full_vs_shared_qkv","dtype":str(dtype),"batch":batch,
                            "max_abs_error":max(float((a-b).abs().max()) for a,b in zip(optimized,reference)),"pass":True})

            adapter=Krea2Adapter(model,negative,cfg,OPS)
            a=adapter(x,t,context)
            b=adapter(x,t,context)
            assert torch.equal(a,b) and adapter.text_fusion_calls == 1
            assert torch.isfinite(a).all() and not torch.equal(a,native)
            adapter.clear()
            results.append({"case":"native_text_fusion_cache_and_nonzero_nag","dtype":str(dtype),"batch":batch,"pass":True})

            # Initialize just the attributes used by the actual apply_model;
            # loading/checkpoint construction is explicitly outside this probe.
            km=KModel.__new__(KModel)
            nn.Module.__init__(km)
            km.diffusion_model=model
            km.computation_dtype=dtype
            km.predictor=Predictor()
            bindings=HostBindings(object,DiT,KModel,object,object,SimpleNamespace(ref_latents=[]),OPS)
            wrapped=NAGModelWrapper(km,Krea2Adapter(model,negative,cfg,OPS),cfg,bindings)
            payload={"input":x.float(),"timestep":t,"c":{"c_crossattn":context.clone(),"transformer_options":{"cond_or_uncond":[0]}},"cond_or_uncond":[0]}
            out=wrapped(km.apply_model,payload)
            assert torch.isfinite(out).all() and out.shape==x.shape
            assert km.predictor.calls == ["input","time","output"]
            assert wrapped.active_calls == 1
            wrapped.adapter.clear()
            results.append({"case":"native_KModel_predictor_preserved","dtype":str(dtype),"batch":batch,"pass":True})
    return {
        "scope":"Actual Forge class/method definitions; reduced random CPU models; backend operator stubs; no pretrained weights or GPU",
        "torch":torch.__version__,"krea_source":krea_hash,"kmodel_source":kmodel_hash,
        "tests":len(results),"passed":sum(row["pass"] for row in results),"results":results,
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--krea-source",required=True,type=Path)
    parser.add_argument("--kmodel-source",required=True,type=Path)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    result=run(args.krea_source,args.kmodel_source)
    text=json.dumps(result,indent=2,ensure_ascii=False)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(text+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
