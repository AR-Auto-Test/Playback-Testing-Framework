package com.ibosoninnov.unitear;

import android.view.Surface;
import java.nio.ByteBuffer;

/* loaded from: classes2.dex */
public class CVLib {
    static {
        System.loadLibrary("cv-lib");
    }

    public final native boolean getTrackStatusJNI();

    public final native float[] getTransformationMatrixJNI();

    public final native void onImageAvailableJNI(int i, int i2, int i3, int i4, ByteBuffer byteBuffer, int i5, ByteBuffer byteBuffer2, int i6, ByteBuffer byteBuffer3, Surface surface, long j, boolean z, boolean z2);

    public final native void patternDetectorInitJNI();

    public final native void patternDetectorSetCameraMatrixJNI(float f2, float f3, float f4, float f5, float f6, float f7);

    public final native boolean patternDetectorSetImageToDetectJNI(long j);
}