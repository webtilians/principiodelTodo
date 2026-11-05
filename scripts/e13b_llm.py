"""E13-B real generative-LLM pilot on completely NEW synthetic LLM worlds.

This module never trains, never knows generator gold when constructing a prompt
and never touches user data or external paid APIs. 8 worlds x (relation,current)
x 4 arms = 64 actual model generations with the SAME prompt and settings.
"""
from __future__ import annotations
import argparse,hashlib,json,re,time
from pathlib import Path

ARMS=("A_CURRENT","B_SETS","C_RELATIONS","D_HYBRID")
FAMILIES=("relation","current")
MODEL="Qwen/Qwen2.5-0.5B-Instruct"
SYSTEM=(
    "Responde preguntas usando EXCLUSIVAMENTE la evidencia suministrada. "
    "No inventes datos. Ten en cuenta las fechas: el hecho factual posterior "
    "sustituye al anterior; una propuesta no es un hecho. "
    "Para relaciones de dos saltos, utiliza solo hechos explícitos. "
    "Devuelve exclusivamente JSON de la forma "
    '{"respuesta":["valor"]}, sin texto adicional. '
    "Si no tienes pruebas, usa una lista vacía."
)

def parse(response):
    raw=response.strip()
    # Do not ask a separate model to score or manually correct its answer.
    try:
        start=raw.index("{")
        end=raw.rindex("}")+1
        payload=json.loads(raw[start:end])
        values=payload["respuesta"]
        if isinstance(values,list) and all(isinstance(v,str) for v in values):
            return sorted(set(values))
    except (ValueError,KeyError,TypeError):
        pass
    return None

def fingerprint(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--folder",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--revision",required=True)
    ap.add_argument("--model",default=MODEL)
    args=ap.parse_args()
    assert len(args.revision)>=12 and args.revision!="main", "model revision must be frozen SHA"
    assert args.model==MODEL, "different LLM is a new experiment"
    assert not args.out.exists(),"LLM output must not overwrite consumed data"
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(2)
    model=AutoModelForCausalLM.from_pretrained(
        args.model,revision=args.revision,torch_dtype=torch.float32).eval()
    tokenizer=AutoTokenizer.from_pretrained(
        args.model,revision=args.revision)
    records=[]
    worldfiles=sorted(args.folder.glob("world-*.json"))
    assert len(worldfiles)==8
    for file in worldfiles:
        w=json.loads(file.read_text())
        for family in FAMILIES:
            for arm in ARMS:
                selected=[x for x in w["arms"][arm]["queries"]
                          if x["budget"]==512 and x["family"]==family]
                assert len(selected)==4
                q=selected[0]
                user=(
                    "EVIDENCIA:\n"+q["context"]+
                    "\n\nPREGUNTA: "+q["query"]+
                    "\nRESPUESTA en JSON:")
                messages=[{"role":"system","content":SYSTEM},
                          {"role":"user","content":user}]
                prompt=tokenizer.apply_chat_template(
                    messages,tokenize=False,add_generation_prompt=True)
                inputs=tokenizer(prompt,return_tensors="pt")
                before=time.perf_counter()
                with torch.no_grad():
                    output=model.generate(**inputs,do_sample=False,
                                          max_new_tokens=48,pad_token_id=tokenizer.eos_token_id)
                completion=tokenizer.decode(
                    output[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True)
                parsed=parse(completion)
                expected=sorted(set(q["expected"]))
                records.append({
                    "world":w["world"],"replica":w["replica"],
                    "seed":w["seed"],"arm":arm,"family":family,
                    "query":q["query"],
                    "context_sha256":hashlib.sha256(q["context"].encode()).hexdigest(),
                    "raw_llm":completion,
                    "parsed":parsed,
                    "expected":expected,
                    "correct":parsed==expected,
                    "prompt_sha256":hashlib.sha256(prompt.encode()).hexdigest(),
                    "input_tokens":int(inputs["input_ids"].shape[1]),
                    "generated_tokens":int(output.shape[1]-inputs["input_ids"].shape[1]),
                    "inference_seconds":round(time.perf_counter()-before,4),
                })
                print("E13B_REAL_LLM",w["replica"],family,arm,
                      "PASS" if parsed==expected else "FAIL",flush=True)
    assert len(records)==64
    summary={}
    for arm in ARMS:
        for f in FAMILIES:
            x=[r for r in records if r["arm"]==arm and r["family"]==f]
            summary[f"{arm}/{f}"]={"correct":sum(r["correct"] for r in x),
                                 "total":8}
    lookups={(r["replica"],r["world"],r["family"],r["arm"]):r
             for r in records}
    diffs={}
    for family in FAMILIES:
        diffs[family]=sum(
            int(lookups[k+(family,"C_RELATIONS")]["correct"])-
            int(lookups[k+(family,"A_CURRENT")]["correct"])
            for k in [(r["replica"],r["world"]) for r in records
                      if r["arm"]=="A_CURRENT" and r["family"]==family])
    g5=diffs["relation"]>=2 and diffs["current"]>=-1 and all(
        not (lookups[k+("relation","A_CURRENT")]["correct"] and
             not lookups[k+("relation","C_RELATIONS")]["correct"])
        for k in [(r["replica"],r["world"]) for r in records
                  if r["arm"]=="A_CURRENT" and r["family"]=="relation"])
    report={
        "status":"E13B_REAL_LLM_EXECUTED",
        "model":args.model,"revision":args.revision,
        "config":{"decode":"greedy","new_tokens":48,"torch_dtype":"float32",
                  "system_prompt":SYSTEM},
        "total_real_model_generations":len(records),
        "scores":summary,
        "paired_C_minus_A_correct":diffs,
        "G5_real_llm":g5,
        "records":records,
    }
    args.out.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print("E13B_LLM_GATE",g5,summary,diffs,flush=True)

if __name__=="__main__":
    main()
