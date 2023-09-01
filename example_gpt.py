import openai
openai.api_key = ''

def main(event):

    prompt = f"Is deze review positief of negatief: " + event

    output = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a Dutch journalist."},
            {"role": "user", "content": prompt}
        ],

        max_tokens=3500
    )

    out_processed = output["choices"][0]["message"]["content"]
    return out_processed


if __name__ == '__main__':
    text_example = 'Het eten was afschuwelijk'
    print(main(text_example))