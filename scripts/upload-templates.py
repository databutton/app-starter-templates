from google.cloud import storage
import os
import tarfile

client = storage.Client(project='databutton')
bucket = client.bucket('databutton-app-templates')
TEMPLATE_PATH = './templates'


def make_tarfile(output_filename, source_dir):
    with tarfile.open(f"{output_filename}.tar.gz", "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))
        return f"{output_filename}.tar.gz"


def prepare():
    # All dirs in the templates folder, ignoring dotfiles
    dirs = [d for d in os.listdir(TEMPLATE_PATH) if not d.startswith('.')]
    return [make_tarfile(dir, f"{TEMPLATE_PATH}/{dir}") for dir in dirs]


def upload(paths):
    print(paths)
    for p in paths:
        blob = bucket.blob(p).upload_from_filename(p)
        print('uploaded', p)


def run():
    upload(prepare())


if __name__ == '__main__':
    run()
