import numpy as np
import torch
import tqdm
import json

DATA_SPLIT ={
    "train": [
        "AimeeNorwich_Child",
        "AimeeNorwich_Flying",
        "AlexanderRoss_GoodbyeBolero",
        "AlexanderRoss_VelvetCurtain",
        "AvaLuna_Waterduct",
        "BigTroubles_Phantom",
        "CroqueMadame_Oil",
        "CroqueMadame_Pilot",
        "DreamersOfTheGhetto_HeavyLove",
        "EthanHein_1930sSynthAndUprightBass",
        "EthanHein_GirlOnABridge",
        "FacesOnFilm_WaitingForGa",
        "FamilyBand_Again",
        "Handel_TornamiAVagheggiar",
        "HeladoNegro_MitadDelMundo",
        "HopAlong_SisterCities",
        "JoelHelander_Definition",
        "JoelHelander_ExcessiveResistancetoChange",
        "JoelHelander_IntheAtticBedroom",
        "KarimDouaidy_Hopscotch",
        "KarimDouaidy_Yatora",
        "LizNelson_Coldwar",
        "LizNelson_ImComingHome",
        "LizNelson_Rainfall",
        "MatthewEntwistle_DontYouEver",
        "Meaxic_TakeAStep",
        "Meaxic_YouListen",
        "Mozart_BesterJungling",
        "MusicDelta_80sRock",
        "MusicDelta_Beatles",
        "MusicDelta_BebopJazz",
        "MusicDelta_Beethoven",
        "MusicDelta_Britpop",
        "MusicDelta_ChineseChaoZhou",
        "MusicDelta_ChineseDrama",
        "MusicDelta_ChineseHenan",
        "MusicDelta_ChineseJiangNan",
        "MusicDelta_ChineseXinJing",
        "MusicDelta_ChineseYaoZu",
        "MusicDelta_CoolJazz",
        "MusicDelta_Country1",
        "MusicDelta_Country2",
        "MusicDelta_Disco",
        "MusicDelta_FreeJazz",
        "MusicDelta_FunkJazz",
        "MusicDelta_GriegTrolltog",
        "MusicDelta_Grunge",
        "MusicDelta_Hendrix",
        "MusicDelta_InTheHalloftheMountainKing",
        "MusicDelta_LatinJazz",
        "MusicDelta_ModalJazz",
        "MusicDelta_Punk",
        "MusicDelta_Reggae",
        "MusicDelta_Rock",
        "MusicDelta_Rockabilly",
        "MusicDelta_Shadows",
        "MusicDelta_SpeedMetal",
        "MusicDelta_Vivaldi",
        "MusicDelta_Zeppelin",
        "PurlingHiss_Lolita",
        "Schumann_Mignon",
        "StevenClark_Bounty",
        "SweetLights_YouLetMeDown",
        "TheDistricts_Vermont",
        "TheScarletBrand_LesFleursDuMal",
        "TheSoSoGlos_Emergency",
        "Wolf_DieBekherte"
    ],
    "valid": [
        "AmarLal_Rest",
        "AmarLal_SpringDay1",
        "BrandonWebster_DontHearAThing",
        "BrandonWebster_YesSirICanFly",
        "ClaraBerryAndWooldog_AirTraffic",
        "ClaraBerryAndWooldog_Boys",
        "ClaraBerryAndWooldog_Stella",
        "ClaraBerryAndWooldog_TheBadGuys",
        "ClaraBerryAndWooldog_WaltzForMyVictims",
        "HezekiahJones_BorrowedHeart",
        "InvisibleFamiliars_DisturbingWildlife",
        "MichaelKropf_AllGoodThings",
        "NightPanther_Fire",
        "SecretMountains_HighHorse",
        "Snowmine_Curfews"
    ],
    "test": [
        "AClassicEducation_NightOwl",
        "Auctioneer_OurFutureFaces",
        "CelestialShore_DieForUs",
        "ChrisJacoby_BoothShotLincoln",
        "ChrisJacoby_PigsFoot",
        "Creepoid_OldTree",
        "Debussy_LenfantProdigue",
        # "MatthewEntwistle_DontYouEver",
        "MatthewEntwistle_FairerHopes",
        "MatthewEntwistle_ImpressionsOfSaturn",
        "MatthewEntwistle_Lontano",
        "MatthewEntwistle_TheArch",
        "MatthewEntwistle_TheFlaxenField",
        "Mozart_DiesBildnis",
        "MusicDelta_FusionJazz",
        "MusicDelta_Gospel",
        "MusicDelta_Pachelbel",
        "MusicDelta_SwingJazz",
        "Phoenix_BrokenPledgeChicagoReel",
        "Phoenix_ColliersDaughter",
        "Phoenix_ElzicsFarewell",
        "Phoenix_LarkOnTheStrandDrummondCastle",
        "Phoenix_ScotchMorris",
        "Phoenix_SeanCaughlinsTheScartaglen",
        "PortStWillow_StayEven",
        "Schubert_Erstarrung",
        "StrandOfOaks_Spacestation"
    ],
}

def hz_to_midi_fn(freqs, ref_freq=440):
    notes = torch.zeros_like(freqs, dtype=torch.float32, device=freqs.device)
    positives = torch.nonzero(freqs)
    notes[positives] = 12 * torch.log2(freqs[positives] / ref_freq) + 69
    return torch.round(notes)

def tensor_to_tuple_string(tensor: torch.Tensor) -> str:
    """
    Convert a torch tensor of shape (n,3) to a formatted string of (time, MIDI) tuples.
    
    Parameters:
        tensor (torch.Tensor): Input tensor with shape (n,3).
    
    Returns:
        str: A string representing a list of tuples with formatted (time, MIDI).
    """
    # Extract first (time) and last (MIDI) columns
    time_values = tensor[:, 0].tolist()
    midi_values = tensor[:, -1].tolist()
    # Format each tuple with time rounded to 2 decimal places and MIDI as an integer
    formatted_tuples = [(f"{t:.4f}", int(m)) for t, m in zip(time_values, midi_values)]
    # Convert list to string
    return str(formatted_tuples)


clip_duration = 10.0
      
data_samples = []

for split in ["test"]:
    # TODO: better to merge the train/valid
    track_names = DATA_SPLIT[split]
    label_files = [
        f"MedleyDB-Melody/melody2/{track_name}_MELODY2.csv" for track_name in track_names
    ]
    audio_files = [
        f"MedleyDB-Melody/audio/{track_name}_MIX.wav"
        for track_name in track_names
    ]
    for idx, label_file in tqdm.tqdm(enumerate(label_files), total=len(label_files), desc=f"Processing {split}"):
        times_labels = torch.Tensor(np.genfromtxt(label_file, delimiter=",")) #(time,2)
        label_notes = hz_to_midi_fn(times_labels[:, 1]).view(
            -1, 1
        ) #(time,1)
        times_labels = torch.hstack((times_labels, label_notes)) #(time,3)
        time_offsets = torch.arange(
            0, times_labels[-1, 0] + clip_duration, clip_duration
        ) #times_labels[-1, 0] is time ends, time_offsets = tensor([  0.,  30.,  60., ...
        intervals = torch.vstack((time_offsets[:-1], time_offsets[1:])) # 0-30, 30-60, ...
        label_invervals = torch.logical_and(
            times_labels[:, :1] > intervals[0], times_labels[:, :1] < intervals[1]
        ).T  # shape (#intervals ?, times TF inside the intervals or not )
        for offset, label_interval in zip(time_offsets, label_invervals):
            data_samples.append({
                "instruction": "please estimate the pitch sequnnce in MIDI number with the timestep of the given audio. The output format should be a list of (float:time (second), int:MIDI number) tuples.",
                "input": f"<|SOA|><AUDIO><|EOA|>",
                "output": tensor_to_tuple_string(times_labels[label_interval]),
                "uuid": "",
                "split": [split if split != "valid" else "dev"],
                "task_type": {"major": ["seq_multi-class"], "minor": ["melody_extraction"]},
                "domain": "music",
                "audio_path": ["data/MedleyDB/"+audio_files[idx]],
                "audio_start": float(offset.cpu()), 
                "audio_end": float(offset.cpu()) + clip_duration,
                "source": "MedleyDB melody2",
                "other": {"tag":"null"}
            })
    
with open(f"CMI_MedleyDB.jsonl", "w") as f:
    for data_sample in data_samples:
        f.write(json.dumps(data_sample) + "\n")
