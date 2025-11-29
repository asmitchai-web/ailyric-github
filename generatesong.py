def generate_song(model, tokenizer, genre, mood, structure, temperature=0.9):

    sections = structure.split("-")  # e.g., V-C-V-C-B-C
    prompt = f"{genre} {mood} <VERSE> "

    output_text = ""

    for sec in sections:
        token = {
            "V": "<VERSE>",
            "C": "<CHORUS>",
            "B": "<BRIDGE>"
        }[sec]

        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        output = model.generate(
            input_ids,
            max_length=200,
            do_sample=True,
            temperature=temperature,
            top_p=0.9
        )

        generated = tokenizer.decode(output[0], skip_special_tokens=True)
        section_text = generated.replace(prompt, "").strip()

        output_text += f"\n\n{token}\n{section_text}\n"
        prompt += section_text + f"\n{token} "

    return output_text

