import time
import cv2
import numpy as np
from mss import mss
import argparse
import pynput.keyboard
from pynput.keyboard import KeyCode
import threading
import pyaudio
import wave
import subprocess
import os


class VideoCapture(threading.Thread):

    def __init__(self, **kwargs):
        super().__init__()
        self.filename = 'video' 
        self.fps = kwargs.get('fps', 30.0)
        self.folder = kwargs['output_dir'] 
        self.monitor = None

        self.extension = '.mp4'
        self.run_flag = False
        self.video_writer = None

        self.start_time = None
        self.frame_count = None
        self.elapsed_time = None

    def run(self):
        self._record()
        self.elapsed_time = time.time() - self.start_time
        self._free()

    def stop(self):
        self.run_flag = False 

    def _record(self):
        self.run_flag = True 
        self.frame_count = 0
        self.elapsed_time = 0
        
        with mss() as sct:
            self.monitor = sct.monitors[0]
            path = os.path.join(self.folder, self.filename + self.extension)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            self.video_writer = cv2.VideoWriter(path, fourcc, self.fps, (self.monitor['width'], self.monitor['height']))

            self.start_time = time.time()
            last_time = 0
            while self.run_flag:
                if time.time() - last_time >= 1./self.fps:
                    last_time = time.time()
                    img = sct.grab(self.monitor)
                    self.video_writer.write(cv2.cvtColor(np.array(img), cv2.COLOR_BGRA2BGR))
                    self.frame_count += 1

    def _free(self):
        self.video_writer.release()
        cv2.destroyAllWindows()
        print('video recording stopped')


class AudioCapture(threading.Thread):
    """
    To record audio set the stereo mixer 
    as the default recording device.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.filename = 'audio'
        self.folder = kwargs['output_dir']

        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 2
        self.RATE = 44100
        self.extension = '.wav'

        self.run_flag = False 
        self.audio_frames = None
        self.audio = None
        self.stream = None 

    def run(self):
        self._record()
        self._free()

    def stop(self):
        self.run_flag = False 

    def _record(self):
        self.run_flag = True 
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=self.FORMAT, channels=self.CHANNELS, rate=self.RATE, input=True, frames_per_buffer=self.CHUNK)
        self.audio_frames = []

        info = self.audio.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')
        device_names = []
        device_id = None
        for i in range(0, numdevices):
            if self.audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels') > 0:
                device_name = self.audio.get_device_info_by_host_api_device_index(0, i).get('name')
                print(f'device {i}:', device_name)
        
                if 'stereomix' in device_name.lower():
                    self.audio.input_device_index = i

        print('selected audio device number:', self.audio.input_device_index)
                    
        while self.run_flag:
            data = self.stream.read(self.CHUNK) 
            self.audio_frames.append(data)

        print('saving recorded audio')
        path = os.path.join(self.folder, self.filename + self.extension)
        wf = wave.open(path, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(self.audio.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(self.audio_frames))
        wf.close()
        
    def _free(self):
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()
        print('audio recording stopped')


def _to_float(x: float):
    return float(f'{x:.2f}')


def merge(video_capture, audio_capture, **kwargs):
    """ 
    https://ffmpeg.org/download.html and add the path of the ffmpeg.exe
    to the environment variable Path
    """
    print('merging video and audio...')
    filename = kwargs['filename']

    try:
        recorded_fps = _to_float(video_capture.frame_count / video_capture.elapsed_time)
        print('recorded fps:', recorded_fps, 'desired fps:', video_capture.fps)
    except:
        with open('info.log', 'r') as fh:
            content = fh.read()
        recorded_fps, video_capture.fps = map(float, content.split(' ')) 

    # if the fps rate was higher/lower than expected, 
    # re-encode it to the expected fps
    audio_file = os.path.join(audio_capture.folder, audio_capture.filename + audio_capture.extension)
    video_file = os.path.join(video_capture.folder, video_capture.filename + video_capture.extension)

    out_file = os.path.join(video_capture.folder, filename + video_capture.extension)

    if abs(recorded_fps - video_capture.fps) >= 0.01:    
        print("re-encoding")
        ffmpeg = os.path.abspath(kwargs['ffmpeg_path'])
        video_file2 = os.path.join(video_capture.folder, video_capture.filename + '2' + video_capture.extension)
        cmd = f"{ffmpeg} -r {recorded_fps} -i {video_file} -pix_fmt yuv420p -r {video_capture.fps} {video_file2}"
        subprocess.call(cmd, shell=True)

        print('muxing')
        cmd = f"{ffmpeg} -ac 2 -channel_layout stereo -i {audio_file} -i {video_file2} -pix_fmt yuv420p {out_file}"
        subprocess.call(cmd, shell=True)
    else:
        print('normal recording and muxing')
        cmd = f"{ffmpeg} -ac 2 -channel_layout stereo -i {audio_file} -i {video_file} -pix_fmt yuv420p {out_file}"
        subprocess.call(cmd, shell=True)

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str)  # final recording result name
    parser.add_argument('--exit_key', type=str, default='q')
    parser.add_argument('--ffmpeg_path', type=str, default='./ffmpeg/bin/ffmpeg.exe')
    parser.add_argument('--output_dir', type=str, default='./output')

    args = parser.parse_args()

    if len(args.exit_key) != 1:
        print('invalid exit key set! exit key will be set to: q')
        args.exit_key = 'q'

    p = args.ffmpeg_path
    if not os.path.exists(p):
        raise AttributeError('ffmpeg not found at given path')

    p = args.output_dir
    if not os.path.exists(p):
        p = os.path.abspath(p)
        answer = input(f'output folder does not exist, want to create one at: {p} (y/n)?')
        if answer.lower() == 'y':
            os.makedirs(p)
        else:
            raise AttributeError('given output folder does not exist')

    kwargs = args.__dict__

    exit_key = KeyCode(char=args.exit_key)

    vc = VideoCapture(**kwargs)
    ac = AudioCapture(**kwargs)

    # capture video and audio:
    vc.start()
    ac.start()

    def on_press(key):
        if key == exit_key:
            print('exit key pressed')
            vc.stop()
            vc.join()
            ac.stop()
            ac.join()
            keylistener.stop()

    keylistener = pynput.keyboard.Listener(on_press=on_press)
    keylistener.start()
    keylistener.join()

    with open('info.log', 'w') as fh:
        recorded_fps = _to_float(vc.frame_count / vc.elapsed_time)
        desired_fps = vc.fps
        print(recorded_fps, desired_fps, file=fh)

    # merge video and audio:
    merge(vc, ac, **kwargs)

