import torch

CONTEXT_NORMS = {
    'mean': [92.1179, 413.7875, 470.5253],
    'mad': [53.4742, 232.5775, 251.1813],
}

state_dict = torch.load(
                "EDM_MODEL_301.weights",
                map_location="cpu",
            )


license_key = "Non-commercial research use only. See LICENSE-MODEL for details."
author = "Denis Sapegin"
year = 2025

del state_dict["buffer"]

new_state_dict = dict()

new_state_dict["state_dict"] = state_dict
new_state_dict["context_norms"] = CONTEXT_NORMS
new_state_dict["license"] = license_key
new_state_dict["author"] = author
new_state_dict["year"] = year

torch.save(
                        new_state_dict,
                        f"licensed_edm_moi_chembl_6_39_final.pt",
                    )

