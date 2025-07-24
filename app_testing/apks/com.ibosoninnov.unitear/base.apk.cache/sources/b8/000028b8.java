package com.google.mediapipe.components;

import android.media.AudioFormat;
import java.nio.ByteBuffer;

/* JADX WARN: Classes with same name are omitted:
  classes2.dex
 */
/* loaded from: base.apk:classes.jar:com/google/mediapipe/components/AudioDataConsumer.class */
public interface AudioDataConsumer {
    void onNewAudioData(ByteBuffer audioData, long timestampMicros, AudioFormat audioFormat);
}