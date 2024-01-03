import pdb
import moviepy.editor as mp
import os
file_list = [
    'a_asian_Santa_Claus.mp4',
    'a_metal_bunny_sitting_on_top_of_a_stack_of_chocolate_cookie.mp4',
    'Beautifully_designed_hyper-realistic_futuristic_electric_vehicle_for_elderly_people_highest_poly_count_highest_contrast_highest_detail_highest_quality_UHD.mp4',
    'Beautifully_designed_hyper-realistic_psychedelic_bee-concept_futuristic_fighter_jet_aircraft_highest_contrast_highest_poly_count_highest_detail_highest_quality_UHD.mp4'
]
method_list = [
    'magic3d',
    'prolific',
    'ours'
]

for file in file_list:

    video1 = mp.VideoFileClip(os.path.join(method_list[0], file))
    video2 = mp.VideoFileClip(os.path.join(method_list[1], file))
    video3 = mp.VideoFileClip(os.path.join(method_list[2], file))

    # 将三个视频水平拼接成一个
    final_video = mp.clips_array([[video1, video2, video3]])

    # 保存拼接后的视频
    final_video.write_videofile(file, codec='libx264')
