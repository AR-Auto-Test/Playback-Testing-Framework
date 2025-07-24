package com.google.firebase.encoders;

/* loaded from: classes2.dex */
public interface ValueEncoderContext {
    ValueEncoderContext add(double d2);

    ValueEncoderContext add(float f2);

    ValueEncoderContext add(int i);

    ValueEncoderContext add(long j);

    ValueEncoderContext add(String str);

    ValueEncoderContext add(boolean z);

    ValueEncoderContext add(byte[] bArr);
}