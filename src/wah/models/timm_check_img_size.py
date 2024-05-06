# TODO: fix code
import timm

import wah

checked = []
need_img_size = []

timm_models = timm.list_models()
NUM_MODELS = len(timm_models)

for i, model_name in enumerate(timm_models):
    try:
        print(
            f"[{i + 1}/{NUM_MODELS}] Checking if {model_name} needs img_size as arg..."
        )

        model = timm.create_model(
            model_name, pretrained=False, in_chans=3, img_size=32
        )
        need_img_size.append(model_name)

    except:
        continue

print("\ndone.")

wah.lst.save_in_txt(need_img_size, "timm_need_img_size")
wah.lst.save_in_txt(checked, "timm_family")
