import os
import requests
import time

papers = [
    {
        "title": "PlantDoc_Dataset_Visual_Plant_Disease_Detection.pdf",
        "url": "https://arxiv.org/pdf/1911.10317"
    },
    {
        "title": "DINOv2_Learning_Robust_Visual_Features_Without_Supervision.pdf",
        "url": "https://arxiv.org/pdf/2304.07193"
    },
    {
        "title": "Swin_Transformer_Hierarchical_Vision_Transformer.pdf",
        "url": "https://arxiv.org/pdf/2103.14030"
    },
    {
        "title": "ConvNeXt_V2_Co-designing_and_Scaling_ConvNets.pdf",
        "url": "https://arxiv.org/pdf/2301.00808"
    },
    {
        "title": "PlantXViT_Explainable_Vision_Transformer_for_Plant_Disease.pdf",
        "url": "https://arxiv.org/pdf/2207.07919"
    }
]

def download_papers(save_dir="Papers"):
    os.makedirs(save_dir, exist_ok=True)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    for paper in papers:
        target_path = os.path.join(save_dir, paper["title"])
        if os.path.exists(target_path) and os.path.getsize(target_path) > 10000:
            print(f"Already exists: {paper['title']}")
            continue

        print(f"Downloading {paper['title']} from {paper['url']}...")
        try:
            resp = requests.get(paper["url"], headers=headers, timeout=30)
            if resp.status_code == 200:
                with open(target_path, "wb") as f:
                    f.write(resp.content)
                print(f"Saved {paper['title']} ({len(resp.content)/(1024*1024):.2f} MB)")
            else:
                print(f"Failed ({resp.status_code}): {paper['url']}")
        except Exception as e:
            print(f"Error downloading {paper['title']}: {e}")
        time.sleep(1)

    print(f"\nAll papers successfully processed in folder '{save_dir}'.")

if __name__ == "__main__":
    download_papers()
