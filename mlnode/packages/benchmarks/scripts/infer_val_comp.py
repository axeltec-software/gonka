import json


filepath_inf = 'infer_resp.json'
filepath_inf_out = 'infer_res.txt'

filepath_val = 'val_resp.json'
filepath_val_out = 'val_res.txt'

filepath_out = 'res.txt'
filepath_out_wrong_tok = 'res_wrong_tok.txt'

data_inf = []
top_decoded_tokens = []
with open(filepath_inf, 'r', encoding='utf-8') as f_inf:
    for line in f_inf:
        try:
            data_inf.append(json.loads(line.strip()))
            break
        except json.JSONDecodeError as e:
            print(f"Skipping malformed line: {line.strip()} - Error: {e}")

if data_inf:
    with open(filepath_inf_out, 'w', encoding='utf-8') as f_inf_out:
        for resp in data_inf:
            prompt_len = resp["usage"]["prompt_tokens"]
            logprobs_inf_resp = resp["choices"][0]["logprobs"]["content"]
            for el in logprobs_inf_resp:
                top_decoded_tokens.append(el["top_logprobs"])
                f_inf_out.write(f"{str(el["top_logprobs"])}\n")



data_val = []
top_prompt_tokens = []
with open(filepath_val, 'r', encoding='utf-8') as f_val:
    for line in f_val:
        try:
            data_val.append(json.loads(line.strip()))
            break
        except json.JSONDecodeError as e:
            print(f"Skipping malformed line: {line.strip()} - Error: {e}")

if data_val:
    with open(filepath_val_out, 'w', encoding='utf-8') as f_val_out:
        for resp in data_val:
            logprobs_val_resp = resp["prompt_logprobs"][prompt_len:]
            for el in logprobs_val_resp:
                top_prompt_tokens.append(el)
                f_val_out.write(f"{str(el)}\n")

k = len(top_decoded_tokens[0])
with open(filepath_out, 'w', encoding='utf-8') as f_out:
    with open(filepath_out_wrong_tok, 'w', encoding='utf-8') as f_out_wt:
        for pair in zip(top_decoded_tokens, top_prompt_tokens):
            for t in range(k): 
                tok = pair[0][t]['token']
            
                try:
                    lp_tok_inf = pair[0][t]['logprob']
                    lp_tok_val = pair[1][tok]['logprob']
                except KeyError:
                    pass

                rel = (1.0 - float(lp_tok_inf / lp_tok_val)) * 100 if lp_tok_val else ""
                
                if tok == list(pair[1])[t]:
                    f_out.write(f"{tok}\t{lp_tok_inf}\t{lp_tok_val}\t{rel:.5}\n")
                else:
                    f_out_wt.write(f"{tok}\t{lp_tok_inf}\t{list(pair[1])[t]}\t{pair[1][list(pair[1])[t]]['logprob']}\n")