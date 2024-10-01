import click

@click.command()
@click.option('--request', prompt='Image prompt', help='Describe the image to generate.')
def generate(request):

    # Note: DALL-E 3 requires version 1.0.0 of the openai-python library or later
    import os
    from openai import AzureOpenAI
    import json
    from IPython.display import Image
    from urllib.request import urlretrieve
    from dotenv import load_dotenv
    import yaml

    load_dotenv()

    api_base = os.environ.get("AZURE_OPENAI_API_BASE", "https://api.openai.com/v1")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", None)

    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=api_base,
        api_key=api_key,
    )

    images_paths = []

    print(f'Generating image based on prompt: {request}')

    from datetime import datetime
    from pathlib import Path
    def gen_filename(prompt):
        ts = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        return f"work/{ts} {prompt}"[:251]

    def write_prompt(request, prompt, filename):
        with open(filename, "w") as f:
            content = {
                "request": request,
                "dalle_prompt": prompt
            }
            yaml.dump(content, f)

    system = """You are a helpful AI assistant specialized in Dalle-3 Image generation."""

    prompt = f"""
    Please generate the best possible Dalle-3 prompt for the following image request: {request}.
    Include only the resulting Dalle-3 prompt in your response.
    """

    print("Generating Dall-e 3 prompt ...")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )

    dalle_prompt = response.choices[0].message.content
    print(f"Dall-e 3 prompt: {dalle_prompt}")

    print("Generating image ...")
    result = client.images.generate(
        model="dall-e-3", # the name of your DALL-E 3 deployment
        prompt=dalle_prompt,
        size="1792x1024",
        n=1
    )

    image_url = json.loads(result.model_dump_json())['data'][0]['url']
    file_path_suffix=gen_filename(request)
    file_path=Path(file_path_suffix + ".png")
    print(f"Saving image to file {file_path} ...")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(image_url, file_path)
    write_prompt(request, dalle_prompt, Path(file_path_suffix + ".yaml"))
    images_paths.append(file_path)
    Image(filename=file_path)

if __name__ == '__main__':
    generate()
