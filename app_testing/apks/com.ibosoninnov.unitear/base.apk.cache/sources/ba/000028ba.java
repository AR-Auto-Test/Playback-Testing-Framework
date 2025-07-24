package com.google.mediapipe.components;

import org.opencv.features2d.AKAZE;

/* JADX WARN: Classes with same name are omitted:
  classes2.dex
 */
/* loaded from: base.apk:classes.jar:com/google/mediapipe/components/AudioDataProducer.class */
public interface AudioDataProducer {
    AKAZE create(int consumer);
}