import os
import imageio
import numpy as np

from .visualization_utils import make_grid, save_numpy_as_video


def video_pad_time(videos):
    nframe = np.max([video.shape[0] for video in videos])
    padded = []
    for video in videos:
        npad = nframe - len(video)
        padded_frame = video[[-1], :, :, :].copy()
        video = np.vstack([video, np.tile(padded_frame, [npad, 1, 1, 1])])
        padded.append(video)
    return np.array(padded)


def make_grid_video_from_numpy(video_array, ncol, output_name='./output.mp4', speedup=1, padding=5, **kwargs):
    videos = []
    for video in video_array:
        if speedup != 1:
            video = video[::speedup]
        videos.append(video)
    videos = video_pad_time(videos)  # N x T x H x W x 3
    grid_frames = []
    for t in range(videos.shape[1]):
        grid_frame = make_grid(videos[:, t], ncol=ncol, padding=padding)
        grid_frames.append(grid_frame)
    save_numpy_as_video(np.array(grid_frames), output_name, **kwargs)


def rearrange_videos(videos, success, success_vid_first, fail_vid_first):
    success = np.array(success)
    rearrange_idx = np.arange(len(success))
    if success_vid_first:
        success_idx = rearrange_idx[success]
        fail_idx = rearrange_idx[np.logical_not(success)]
        videos = np.concatenate([videos[success_idx], videos[fail_idx]], axis=0)
        rearrange_idx = np.concatenate([success_idx, fail_idx], axis=0)
    if fail_vid_first:
        success_idx = rearrange_idx[success]
        fail_idx = rearrange_idx[np.logical_not(success)]
        videos = np.concatenate([videos[fail_idx], videos[success_idx]], axis=0)
        rearrange_idx = np.concatenate([fail_idx, success_idx], axis=0)
    return videos, rearrange_idx


def render_done_to_boundary(frame, success, color=(0, 255, 0)):
    """
    If done, render a color boundary to the frame.
    Args:
        frame: (b, c, h, w)
        success: (b, 1)
        color: rgb value to illustrate success, default: (0, 255, 0)
    """
    if any(success):
        b, c, h, w = frame.shape
        color = np.array(color, dtype=frame.dtype)[None, :, None, None]
        boundary = int(min(h, w) * 0.015)
        frame[success, :, :boundary, :] = color
        frame[success, :, -boundary:, :] = color
        frame[success, :, :, :boundary] = color
        frame[success, :, :, -boundary:] = color
    return frame


class VideoWriter:
    def __init__(self, video_path, save_video=False, fps=30, single_video=False):
        self.video_path = video_path
        self.save_video = save_video
        self.fps = fps
        self.image_buffer = {}
        self.last_images = {}
        self.single_video = single_video
        self.success = None
        self.env_description = None
        self.num_env_rollout = None
        self.task_idx = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # self.save()
        pass

    def reset(self):
        if self.save_video:
            self.last_images = {}
            self.image_buffer = {}
            self.success = None
            self.env_description = None
            self.num_env_rollout = None
            self.task_idx = None

    def append_image(self, img, idx=0):
        """Directly append an image to the video."""
        if self.save_video:
            if idx not in self.image_buffer:
                self.image_buffer[idx] = []
            self.image_buffer[idx].append(img)

    def append_obs(self, obs, done, idx=0, camera_name="agentview_image", boundary=True):
        """Append a camera observation to the video."""
        if self.save_video:
            if idx not in self.image_buffer:
                self.image_buffer[idx] = []
            if idx not in self.last_images:
                self.last_images[idx] = None

            if not done:
                self.image_buffer[idx].append(obs[camera_name][::-1])
            else:
                if self.last_images[idx] is None:
                    self.last_images[idx] = obs[camera_name][::-1]
                original_image = np.copy(self.last_images[idx])

                if boundary:
                    border_color = [0, 255, 0]      # green
                    border_thickness = 3  
                    original_image[:border_thickness, :, :] = border_color  
                    original_image[-border_thickness:, :, :] = border_color 
                    original_image[:, :border_thickness, :] = border_color 
                    original_image[:, -border_thickness:, :] = border_color  
                else:
                    blank_image = np.ones_like(original_image) * 128
                    blank_image[:, :, 0] = 0
                    blank_image[:, :, -1] = 0
                    transparency = 0.7
                    original_image = (
                        original_image * (1 - transparency) + blank_image * transparency
                    )

                self.image_buffer[idx].append(original_image.astype(np.uint8))

    def append_vector_obs(self, obs, dones, camera_name="agentview_image"):
        if self.save_video:
            for i in range(len(obs)):
                self.append_obs(obs[i], dones[i], i, camera_name)
    
    def get_last_info(self, num_env_rollout, dones, env_description, task_idx):    
        self.success = dones
        self.env_description = env_description.split('/')[-1]
        self.num_env_rollout = num_env_rollout
        self.task_idx = task_idx

    def save(self):
        if self.save_video:
            env_description = str(self.task_idx) + '.' + self.env_description
            final_video_path = os.path.join(self.video_path, env_description)
            os.makedirs(final_video_path, exist_ok=True)
            if self.single_video:
                video_name = os.path.join(final_video_path, f"video.mp4")
                video_writer = imageio.get_writer(video_name, fps=self.fps)
                for idx in self.image_buffer.keys():
                    for im in self.image_buffer[idx]:
                        video_writer.append_data(im)
                video_writer.close()
            else:
                for idx, suffix in zip(self.image_buffer.keys(), self.success):
                    video_name = os.path.join(final_video_path, f"rollout{self.num_env_rollout}_env{idx}_{str(suffix)}.mp4")
                    video_writer = imageio.get_writer(video_name, fps=self.fps)
                    for im in self.image_buffer[idx]:
                        video_writer.append_data(im)
                    video_writer.close()
            print(f"Saved videos to {os.path.join(final_video_path)}.")

    