import cv2
import numpy as np
import sys
from PyQt5 import QtWidgets as qtw
from PyQt5 import QtGui as qtg
from PyQt5 import QtCore as qtc
from frame_to_skeleton import Pose_Estimation
from encoder import ST_GCN_Encoder
from renderer import Kpts_Renderer

class WebcamThread(qtc.QThread):
    # send self.frame_kpts_list_3d to Encoder
    send_webcam_kpts = qtc.pyqtSignal(np.ndarray, np.ndarray, np.ndarray)

    change_pixmap_signal = qtc.pyqtSignal(np.ndarray)

    # send rgb_image to pose thread
    send_rgb_image = qtc.pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.frame_idx = 1
        self.fps = 10
        self.frame_kpts_list_2d = []
        self.frame_kpts_list_3d = []
        self.frame_and_kpts = []
        self.pose_estimation_engine = Pose_Estimation(self.fps)
        self.renderer = Kpts_Renderer(kpts_type="mediapipe")
        self._run_flag = True
        self.run_encoding = False
        self.cap = cv2.VideoCapture(0)
        # Not sure if this can work
        self.cap.set(cv2.CAP_PROP_FPS, 30)

    def run(self):
        # capture from video
        if not self.cap.isOpened():
            exit()

        while self._run_flag:
            ret, frame = self.cap.read()
            if ret:
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                #rgb_image = cv2.flip(rgb_image, 1)

                # pose estimation
                kpts_pose, kpts_pose_17, kpts_3d = self.pose_estimation_engine.img_to_kpts(rgb_image)
                if kpts_pose is not None:
                    self.frame_kpts_list_2d.append(kpts_pose_17)
                    self.frame_kpts_list_3d.append(kpts_3d)
                    rgb_image = self.renderer.draw_landmarks_on_video(rgb_image, kpts_pose)
                    self.frame_and_kpts.append(rgb_image)

                if len(self.frame_kpts_list_3d) == 10:
                    self.run_encoding = True
                    self.send_webcam_kpts.emit(np.array(self.frame_kpts_list_3d, dtype=np.float32), np.array(self.frame_kpts_list_2d, dtype=np.float32), np.array(self.frame_and_kpts, dtype=np.float32))
                    self.frame_kpts_list_2d = []
                    self.frame_kpts_list_3d = []
                    self.frame_and_kpts = []
                    # print("Webcam",self.frame_idx-10, self.frame_idx)
                
                self.change_pixmap_signal.emit(rgb_image)
                self.frame_idx += 1
                
            # qtc.QThread.msleep(int((1/self.fps)*1000))

    # @qtc.pyqtSlot(float)
    # def receive_encoding_finish_signal(self, similarity):
    #     self.run_encoding = False

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        # shut down capture system
        self.cap.release()
        self.quit()
        self.wait()

class VideoThread(qtc.QThread):
    # Emit a signal when cv2 reads a new frame
    # The parameter is a np array (i.e., image)
    change_pixmap_signal = qtc.pyqtSignal(np.ndarray)

    # Send self.teacher_kpts_3d to Encoder
    send_video_kpts = qtc.pyqtSignal(np.ndarray, np.ndarray)

    # Emit a signal to indicate that the video has finished playing
    # This signal is sent to the main thread and Encoder
    video_is_over = qtc.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.frame_idx = 1
        self.fps = 10
        # shape: (261, 133, 3)
        self.teacher_kpts_2d = np.load("./teacher_data/f07_v00_h16_00649_fps10_2d.npz")["keypoints"]
        # shape: (261, 17, 4)
        self.teacher_kpts_3d = np.load("./teacher_data/f07_v00_h16_00649_fps10_3d.npz")["keypoints"]
        self.renderer = Kpts_Renderer(kpts_type="vitpose")
        self._run_flag = True
        # or t_pose
        self.cap = cv2.VideoCapture("./teacher_data/07_00_16_0527.mp4")
        self.pause_video = True
        self.replay = True
        self.frame_and_kpts = []

    def run(self):
        # capture from video
        while self._run_flag:
            if self.replay:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                # Read one frame first (as background)
                ret, frame = self.cap.read()
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_image = self.renderer.draw_landmarks_on_video(rgb_image, self.teacher_kpts_2d[self.frame_idx-1])
                self.change_pixmap_signal.emit(rgb_image)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.replay = False
            
            if not self.pause_video:
                ret, frame = self.cap.read()
                if ret:
                    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb_image = self.renderer.draw_landmarks_on_video(rgb_image, self.teacher_kpts_2d[self.frame_idx-1])
                    # rgb_image = cv2.flip(rgb_image, 1)

                    self.frame_and_kpts.append(rgb_image)

                    if self.frame_idx % 10 == 0:
                        self.send_video_kpts.emit(self.teacher_kpts_3d[self.frame_idx-10:self.frame_idx], np.array(self.frame_and_kpts, dtype=np.float32))
                        self.frame_and_kpts = []
                        # print("Video",self.frame_idx-10, self.frame_idx)

                    self.change_pixmap_signal.emit(rgb_image)
                    self.frame_idx += 1
                else:
                    self.video_is_over.emit()
                    self.pause_video = True
            
            qtc.QThread.msleep(int((1/9)*1000))

    @qtc.pyqtSlot()
    def start_play(self):
        self.pause_video = False
        

    def stop(self):
        self._run_flag = False
        # shut down capture system
        self.cap.release()
        self.quit()
        self.wait()


class Encoder(qtc.QThread):
    # Signal emitted after receiving kpts from the webcam and the video
    all_signals_received = qtc.pyqtSignal()
    # Signal emitted after encoding is finished
    task_finished = qtc.pyqtSignal(float)

    # After performing the preparatory pose, start a 3-second countdown before playing the video
    # This signal should be sent to the Main Thread
    t_pose_checked = qtc.pyqtSignal()

    # Send image to write thread to save clip video
    write_clip_signal = qtc.pyqtSignal(np.ndarray, np.ndarray, float)


    def __init__(self):
        super().__init__()
        self.webcam_kpts = None
        self.webcam_kpts_2d = None
        self.video_kpts = None
        self.webcam_imgs = None
        self.video_imgs = None
        self.webcam_kpts_received = False
        self.video_kpts_received = False
        self.running = True
        self.stgcn_encoder = ST_GCN_Encoder()
        #self.t_pose = np.load("./teacher_data/t_pose.npz")["keypoints"]
        self.t_pose = np.repeat(np.load("./teacher_data/f07_v00_h16_00649_fps10_3d.npz")["keypoints"][:1], 10, axis=0)
        self.playing_video = False

        
    
    @qtc.pyqtSlot(np.ndarray, np.ndarray, np.ndarray)
    def receive_webcam_kpts(self, kpts, kpts_2d, imgs):
        self.webcam_kpts = kpts
        self.webcam_kpts_2d = kpts_2d
        self.webcam_imgs = imgs
        self.webcam_kpts_received = True

    @qtc.pyqtSlot(np.ndarray, np.ndarray)
    def receive_video_kpts(self, kpts, imgs):
        self.video_kpts = kpts
        self.video_imgs = imgs
        self.video_kpts_received = True

    @qtc.pyqtSlot()
    def receive_beat(self):
        #print(self.webcam_kpts_received, self.video_kpts_received)
        if not self.playing_video:
            if self.webcam_kpts_received:
                self.process_task(self.webcam_kpts, self.webcam_kpts_2d, self.t_pose, None, None)
                self.reset()
        else:
            if self.webcam_kpts_received and self.video_kpts_received:
                self.process_task(self.webcam_kpts, self.webcam_kpts_2d, self.video_kpts, self.webcam_imgs, self.video_imgs)
                self.reset()
        

    def reset(self):
        self.webcam_kpts = None
        self.webcam_kpts_2d = None
        self.video_kpts = None
        self.webcam_imgs = None
        self.video_imgs = None
        self.webcam_kpts_received = False
        self.video_kpts_received = False


    def run(self):
        # Keep the thread running
        while self.running:
            self.msleep(100)

    def process_task(self, webcam_kpts, webcam_kpts_2d, video_kpts, webcam_imgs, video_imgs):
        similarity = self.stgcn_encoder.start_encoding(webcam_kpts, webcam_kpts_2d, video_kpts)
        if not self.playing_video:
            # performing the preparatory pose
            if similarity > 0.75:
                self.playing_video = True
                self.t_pose_checked.emit()
        else:
            self.write_clip_signal.emit(webcam_imgs, video_imgs, similarity)
        # print("score:",similarity)
        self.task_finished.emit(similarity)
        

    def stop(self):
        self.running = False


class WriteClipThread(qtc.QThread):
    # send clip_dict to main thread
    clip_dict_signal = qtc.pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.clip_dict = {"good":[], "ok":[], "bad":[]}
        self.good_count = 0
        self.ok_count = 0
        self.bad_count = 0
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.fps = 10
        self.img_space = 50
        self.lock = qtc.QMutex()
    
    def pad_image(image, target_height):
        original_height, original_width = image.shape[:2]
        total_padding = target_height - original_height

        if total_padding < 0:
            raise ValueError("Target height must be greater than or equal to original height.")

        top_padding = total_padding // 2
        bottom_padding = total_padding - top_padding

        padded_image = np.pad(image, ((top_padding, bottom_padding), (0, 0), (0, 0)), mode='constant', constant_values=255)

        return padded_image


    @qtc.pyqtSlot(np.ndarray, np.ndarray, float)
    def process_task(self, webcam_imgs, video_imgs, similarity):
        self.lock.lock()
        conbined_img_height = max(webcam_imgs.shape[1], video_imgs.shape[1])
        webcam_img_width = webcam_imgs.shape[2]
        video_img_width = video_imgs.shape[2]
        conbined_img_width = webcam_img_width + video_img_width + self.img_space
        
        if similarity > 0.85:
            output_writer = cv2.VideoWriter(f"./clip_video/good/{self.good_count:03}.mp4", self.fourcc, self.fps, (conbined_img_width, conbined_img_height))
            self.clip_dict["good"].append({"clip_name": f"./clip_video/good/{self.good_count:03}.mp4", "similarity": similarity})
            self.good_count += 1
        elif similarity > 0.65:
            output_writer = cv2.VideoWriter(f"./clip_video/ok/{self.ok_count:03}.mp4", self.fourcc, self.fps, (conbined_img_width, conbined_img_height))
            self.clip_dict["ok"].append({"clip_name": f"./clip_video/ok/{self.ok_count:03}.mp4", "similarity": similarity})
            self.ok_count += 1
        else:
            output_writer = cv2.VideoWriter(f"./clip_video/bad/{self.bad_count:03}.mp4", self.fourcc, self.fps, (conbined_img_width, conbined_img_height))
            self.clip_dict["bad"].append({"clip_name": f"./clip_video/bad/{self.bad_count:03}.mp4", "similarity": similarity})
            self.bad_count += 1
    
        for f in range(len(webcam_imgs)):
            new_image = np.ones((conbined_img_height, conbined_img_width, 3), dtype=np.uint8) * 255
            new_image[:, :video_img_width, :] = video_imgs[f]
            new_image[:, video_img_width+self.img_space:video_img_width+self.img_space+webcam_img_width, :] = webcam_imgs[f]
            output_writer.write(cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR))
        
        output_writer.release()
        self.lock.unlock()

    def send_clip_dict(self):
        self.clip_dict_signal.emit(self.clip_dict)


class ShiningLabel(qtw.QLabel):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setAlignment(qtc.Qt.AlignCenter)
        self.setFont(qtg.QFont('Impact', 50))
        self.setStyleSheet("color: white; background-color: rgba(0,0,0,0);")

        # Set up shadow effect
        self.shadow_effect = qtw.QGraphicsDropShadowEffect(self)
        self.shadow_effect.setBlurRadius(20)
        self.shadow_effect.setColor(qtg.QColor(255, 255, 0))
        self.shadow_effect.setOffset(0, 0)
        self.setGraphicsEffect(self.shadow_effect)

    def set_shadow_color(self, color):
        self.shadow_effect.setColor(color)

class ShiningLabel_Container(qtw.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Container widget for the label
        self.container = qtw.QWidget(self)
        self.container.setStyleSheet("background-color: rgba(0,0,0,0);")
        self.label = ShiningLabel("", self.container)
        layout = qtw.QVBoxLayout(self.container)
        layout.addWidget(self.label)

        # Set up opacity effect
        self.opacity_effect = qtw.QGraphicsOpacityEffect(self.container)
        self.container.setGraphicsEffect(self.opacity_effect)
        
        # Initially invisible
        self.container.setVisible(False)

        # Set up the animations
        self.fadeInAnimation = qtc.QPropertyAnimation(self.opacity_effect, b'opacity')
        self.fadeInAnimation.setDuration(400)
        self.fadeInAnimation.setStartValue(0.0)
        self.fadeInAnimation.setEndValue(1.0)

        self.fadeOutAnimation = qtc.QPropertyAnimation(self.opacity_effect, b'opacity')
        self.fadeOutAnimation.setDuration(200)
        self.fadeOutAnimation.setStartValue(1.0)
        self.fadeOutAnimation.setEndValue(0.0)

        # Combine animations into a sequence
        self.animationGroup = qtc.QSequentialAnimationGroup(self)
        self.animationGroup.addAnimation(self.fadeInAnimation)
        self.animationGroup.addPause(400)
        self.animationGroup.addAnimation(self.fadeOutAnimation)

        # Layout for this container
        main_layout = qtw.QVBoxLayout(self)
        main_layout.addWidget(self.container)

    def flash(self):
        self.container.setVisible(True)
        self.animationGroup.start()
        self.animationGroup.finished.connect(lambda: self.container.setVisible(False))

    def setLabelText(self, text, shadow_color=None):
        self.label.setText(text)
        if shadow_color:
            self.label.set_shadow_color(shadow_color)
        self.flash()

class MainWindow(qtw.QWidget):
    beat_signal = qtc.pyqtSignal()
    start_play_video = qtc.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.total_clip = 0
        self.total_score = 0
        self.clip_dict = None
        self.playing_video = False
        self.count = 3
        self.setWindowTitle("Model Demo")
        self.screen = qtw.QDesktopWidget().screenGeometry()
        self.display_width = self.screen.width()
        self.display_height = self.screen.height()
        self.setGeometry(0, 0, self.display_width, self.display_height)

        self.setStyleSheet("background-color: rgb(226, 225, 224);")

        # create the label that holds the video frame
        self.video_label = qtw.QLabel(self)
        self.video_label.resize(int(2*self.display_width/3), self.display_height)
        self.video_label.setAlignment(qtc.Qt.AlignCenter)

        # create the label that holds the webcam frame
        self.webcam_label = qtw.QLabel(self)
        self.webcam_label.resize(int(self.display_width/3), self.display_height)
        self.webcam_label.setAlignment(qtc.Qt.AlignCenter)

        # create a vertical box layout and add the two labels
        video_layout_h = qtw.QHBoxLayout()
        video_layout_h.addWidget(self.video_label)
        video_layout_h.addWidget(self.webcam_label)
        # set the vbox layout as the widgets layout
        self.video_layout = qtw.QVBoxLayout()
        self.video_layout.addLayout(video_layout_h)
        self.setLayout(self.video_layout)


        # # create a vertical box layout and add the two labels
        # self.video_layout = qtw.QVBoxLayout()
        # self.video_layout.addWidget(self.video_label)
        # # set the vbox layout as the widgets layout
        # self.setLayout(self.video_layout)

        # mask
        self.overlay_label = qtw.QLabel(self)
        self.overlay_label.setGeometry(0, 0, self.display_width, self.display_height)
        self.overlay_label.setStyleSheet("background-color: rgba(0, 0, 0, 0); color: white")
        self.overlay_label.setAlignment(qtc.Qt.AlignCenter)
        self.overlay_label.setText("")
        self.overlay_label.setFont(qtg.QFont('Impact', 60))

        self.label = ShiningLabel_Container(self.video_label)
        self.label.setGeometry(400, 50, 400, 400)

        # create the video capture thread
        self.webcam_thread = WebcamThread()
        # create the encoder thread
        self.encoder_thread = Encoder()
        # create the video capture thread
        self.video_thread = VideoThread()
        # create write clip thread
        self.write_clip_thread = WriteClipThread()

        # Send webcam_kpts to the encoder thread
        self.webcam_thread.send_webcam_kpts.connect(self.encoder_thread.receive_webcam_kpts)
        
        # Send the beat signal to the encoder (which will then compute similarity)
        self.beat_signal.connect(self.encoder_thread.receive_beat)

        # Send the start playback signal to the video thread
        self.start_play_video.connect(self.video_thread.start_play)

        # Send the computed similarity to the main thread
        self.encoder_thread.task_finished.connect(self.update_score_label)

        # Send video_kpts to the encoder thread
        self.video_thread.send_video_kpts.connect(self.encoder_thread.receive_video_kpts)
        
        # Send video images to the main thread
        self.video_thread.change_pixmap_signal.connect(self.update_video_frame)
        self.webcam_thread.change_pixmap_signal.connect(self.update_webcam_frame)
        
        # Detect the preparatory pose and send a signal to the start label
        self.encoder_thread.t_pose_checked.connect(self.update_start_label)

        # Send to encoder
        self.video_thread.video_is_over.connect(self.write_clip_thread.send_clip_dict)

        # Send to the main thread
        self.video_thread.video_is_over.connect(self.video_page_fadeout_effect)
        
        # Write video
        self.encoder_thread.write_clip_signal.connect(self.write_clip_thread.process_task)

        # Save clip_dict received from write thread
        self.write_clip_thread.clip_dict_signal.connect(self.update_clip_dict)

        # Start the threads
        self.webcam_thread.start()
        self.video_thread.start()

        # Independent timer for beat
        self.beat_timer = qtc.QTimer(self)
        self.beat_timer.timeout.connect(self.eavaluation)
        self.beat_timer.start(1500)

    @qtc.pyqtSlot()
    def eavaluation(self):
        self.beat_signal.emit()

    @qtc.pyqtSlot(float)
    def update_score_label(self, score):     
        if self.playing_video:
            self.total_clip += 1
            self.total_score += score
            if score > 0.85:
                self.label.setLabelText("GOOD", qtg.QColor(0, 255, 0))
            elif score > 0.65:
                self.label.setLabelText("OK", qtg.QColor(0, 0, 255))
            else:
                self.label.setLabelText("BAD", qtg.QColor(255, 0, 0))

    @qtc.pyqtSlot(dict)
    def update_clip_dict(self, clip_dict):
        self.clip_dict = clip_dict

    # override
    def closeEvent(self, event):
        super().closeEvent(event)
    
    def fade_out_overlay(self):
        self.opacity_effect = qtw.QGraphicsOpacityEffect()
        self.overlay_label.setGraphicsEffect(self.opacity_effect)
        self.animation = qtc.QPropertyAnimation(self.opacity_effect, b"opacity")
        self.animation.setDuration(2000)
        self.animation.setStartValue(1)
        self.animation.setEndValue(0)
        self.animation.start()


    def update_start_label(self):
        self.countdown_timer = qtc.QTimer(self)
        self.countdown_timer.timeout.connect(self.countdown_to_start)
        self.countdown_timer.start(1000)

    def countdown_to_start(self):
        if self.count > 0:
            self.overlay_label.setText(str(self.count))
            self.count -= 1
        else:
            self.countdown_timer.stop()
            self.overlay_label.setText("Go!")
            self.fade_out_overlay()
            qtc.QTimer.singleShot(3000, self.emit_start_play_video)
    
    def emit_start_play_video(self):
        self.playing_video = True
        self.start_play_video.emit()

    def video_page_fadeout_effect(self):
        self.beat_timer.stop()
        self.encoder_thread.stop()
        self.webcam_thread.stop()
        self.video_thread.stop()
        qtc.QTimer.singleShot(2000, self.video_page_fadeout_effect_2)

    def video_page_fadeout_effect_2(self):
        self.opacity_effect_1 = qtw.QGraphicsOpacityEffect()
        self.opacity_effect_2 = qtw.QGraphicsOpacityEffect()
        self.video_label.setGraphicsEffect(self.opacity_effect_1)
        self.webcam_label.setGraphicsEffect(self.opacity_effect_2)
        self.animation = qtc.QPropertyAnimation(self.opacity_effect_1, b"opacity")
        self.animation.setDuration(2000)
        self.animation.setStartValue(1)
        self.animation.setEndValue(0)
        self.animation.finished.connect(self.show_score_page)
        self.animation.start()
        self.animation2 = qtc.QPropertyAnimation(self.opacity_effect_2, b"opacity")
        self.animation2.setDuration(2000)
        self.animation2.setStartValue(1)
        self.animation2.setEndValue(0)
        # self.animation2.finished.connect(self.show_score_page)
        self.animation2.start()

    def show_score_page(self):
        # for i in reversed(range(self.video_layout.count())): 
        #     self.video_layout.itemAt(i).widget().setParent(None)
        while self.video_layout.count():
            item = self.video_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        spacer_top = qtw.QSpacerItem(20, 40, qtw.QSizePolicy.Minimum, qtw.QSizePolicy.Expanding)
        self.video_layout.addItem(spacer_top)

        self.score_label = qtw.QLabel(f"Your Score: {int(round(self.total_score/self.total_clip, 2)*100)}", self)
        
        self.score_label.setFont(qtg.QFont('Impact', 50))
        self.score_label.setAlignment(qtc.Qt.AlignCenter)
        self.score_label.setStyleSheet("color: rgb(80,79,72);")
        self.video_layout.addWidget(self.score_label)

        spacer_middle = qtw.QSpacerItem(20, 40, qtw.QSizePolicy.Minimum, qtw.QSizePolicy.Expanding)
        self.video_layout.addItem(spacer_middle)

        button_layout = qtw.QHBoxLayout()
        
        self.good_button = qtw.QPushButton(f"Good: {len(self.clip_dict['good'])}", self)
        self.good_button.setFont(qtg.QFont('Impact', 50))
        self.good_button.setMinimumSize(100, 50)
        self.good_button.clicked.connect(lambda: self.show_clip_dialog("Good"))
        self.good_button.setStyleSheet("background-color: rgb(144, 238, 144); color:rgb(72,98,75);")
        button_layout.addWidget(self.good_button)        

        self.ok_button = qtw.QPushButton(f"OK: {len(self.clip_dict['ok'])}", self)
        self.ok_button.setFont(qtg.QFont('Impact', 50))
        self.ok_button.setMinimumSize(100, 50)
        self.ok_button.clicked.connect(lambda: self.show_clip_dialog("OK"))
        self.ok_button.setStyleSheet("background-color: rgb(173, 216, 230); color: rgb(72,91,119);")
        button_layout.addWidget(self.ok_button)

        self.bad_button = qtw.QPushButton(f"Bad: {len(self.clip_dict['bad'])}", self)
        self.bad_button.setFont(qtg.QFont('Impact', 50))
        self.bad_button.setMinimumSize(100, 50) 
        self.bad_button.clicked.connect(lambda: self.show_clip_dialog("Bad"))
        self.bad_button.setStyleSheet("background-color: rgb(255, 182, 193); color:rgb(119,81,72);")
        button_layout.addWidget(self.bad_button)
        button_container = qtw.QWidget()
        button_container.setLayout(button_layout)

        self.video_layout.addWidget(button_container)

        spacer_bottom = qtw.QSpacerItem(20, 40, qtw.QSizePolicy.Minimum, qtw.QSizePolicy.Expanding)
        self.video_layout.addItem(spacer_bottom)

        self.good_button.setEnabled(len(self.clip_dict["good"]) > 0)
        self.ok_button.setEnabled(len(self.clip_dict["ok"]) > 0)
        self.bad_button.setEnabled(len(self.clip_dict["bad"]) > 0)

    def show_clip_dialog(self, rating):
        dialog = ClipDialog(rating, self.clip_dict, self)
        dialog.exec_()

    # def resizeEvent(self, event):
    #   print('resize')
    #   width, height = event.size().width(), event.size().height()
    #   print(width, height)

    # def get_window_size(self):
    #     self.display_width = self.size().width()
    #     self.display_height = self.size().height()
    #     print(f"Display width: {self.display_width}, Display height: {self.display_height}")


    @qtc.pyqtSlot(np.ndarray)
    def update_video_frame(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_to_qt_video(cv_img)
        self.video_label.setPixmap(qt_img)

    @qtc.pyqtSlot(np.ndarray)
    def update_webcam_frame(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_to_qt_webcam(cv_img)
        self.webcam_label.setPixmap(qt_img)

    def convert_cv_to_qt_video(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        cv_img = cv2.flip(cv_img, 1)
        h, w, ch = cv_img.shape
        # bytes_per_line = ch * w
        # convert_to_Qt_format = qtg.QImage(rgb_image.data, w, h, bytes_per_line, qtg.QImage.Format_RGB888)
        qt_img = qtg.QImage(cv_img.data, w, h, qtg.QImage.Format_RGB888)
        #scaled_qt_img = qt_img.scaled(self.display_width, self.display_height, qtc.Qt.KeepAspectRatio)
        scaled_qt_img = qt_img.scaled(int(2*self.display_width/3), self.display_height, qtc.Qt.KeepAspectRatio)
        return qtg.QPixmap.fromImage(scaled_qt_img)
    
    def convert_cv_to_qt_webcam(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        cv_img = cv2.flip(cv_img, 1)
        h, w, ch = cv_img.shape
        # bytes_per_line = ch * w
        # convert_to_Qt_format = qtg.QImage(rgb_image.data, w, h, bytes_per_line, qtg.QImage.Format_RGB888)
        qt_img = qtg.QImage(cv_img.data, w, h, qtg.QImage.Format_RGB888)
        #scaled_qt_img = qt_img.scaled(self.display_width, self.display_height, qtc.Qt.KeepAspectRatio)
        scaled_qt_img = qt_img.scaled(int(self.display_width/3), self.display_height, qtc.Qt.KeepAspectRatio)
        return qtg.QPixmap.fromImage(scaled_qt_img)


class ClipDialog(qtw.QDialog):
    def __init__(self, rating, clips, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Clips for {rating}")
        self.setGeometry(150, 150, 1920, 1080)
        self.center()
        self.clips = clips

        self.layout = qtw.QVBoxLayout()

        self.score_label = qtw.QLabel("Score: ", self)
        self.score_label.setAlignment(qtc.Qt.AlignCenter)
        self.score_label.setFont(qtg.QFont('Impact', 50))
        self.score_label.setStyleSheet("color: rgb(80,79,72);")
        self.layout.addWidget(self.score_label)

        self.clip_label = qtw.QLabel(self)
        self.clip_label.setAlignment(qtc.Qt.AlignCenter)
        self.layout.addWidget(self.clip_label)

        button_layout = qtw.QHBoxLayout()
        self.prev_button = qtw.QPushButton("<", self)
        self.prev_button.clicked.connect(self.show_prev_clip)
        button_layout.addWidget(self.prev_button)

        self.next_button = qtw.QPushButton(">", self)
        self.next_button.clicked.connect(self.show_next_clip)
        button_layout.addWidget(self.next_button)

        self.layout.addLayout(button_layout)

        self.setLayout(self.layout)

        self.current_clip = 0
        self.current_clip_type = rating.lower()
        self.cap = None
        self.timer = qtc.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.show_clip()

    def center(self):
        screen = qtw.QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move((screen.width() - size.width()) // 2, (screen.height() - size.height()) // 2)

    def show_clip(self):
        self.score_label.setText("Clip "+str(self.current_clip)+", Score: "+str(round(self.clips[self.current_clip_type][self.current_clip]["similarity"], 2)))
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.clips[self.current_clip_type][self.current_clip]["clip_name"])
        self.timer.start(100)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            image = qtg.QImage(frame.data, width, height, bytes_per_line, qtg.QImage.Format_RGB888)
            image = image.scaled(1920, 1080, qtc.Qt.KeepAspectRatio)
            self.clip_label.setPixmap(qtg.QPixmap.fromImage(image))
        else:
            self.timer.stop()

    def show_prev_clip(self):
        self.current_clip = (self.current_clip - 1) % len(self.clips[self.current_clip_type])
        self.show_clip()

    def show_next_clip(self):
        self.current_clip = (self.current_clip + 1) % len(self.clips[self.current_clip_type])
        self.show_clip()

    def closeEvent(self, event):
        if self.cap is not None:
            self.cap.release()
        self.timer.stop()
        event.accept()

if __name__=="__main__":
    app = qtw.QApplication(sys.argv)
    main_window = MainWindow()
    main_window.showMaximized() 
    sys.exit(app.exec_())