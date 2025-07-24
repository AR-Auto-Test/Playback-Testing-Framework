package org.opencv.features2d;

/* loaded from: classes2.dex */
public class FlannBasedMatcher extends DescriptorMatcher {
    public FlannBasedMatcher(long j) {
        super(j);
    }

    private static native long FlannBasedMatcher_0();

    public static FlannBasedMatcher __fromPtr__(long j) {
        return new FlannBasedMatcher(j);
    }

    public static FlannBasedMatcher create() {
        return __fromPtr__(create_0());
    }

    private static native long create_0();

    private static native void delete(long j);

    @Override // org.opencv.features2d.DescriptorMatcher, org.opencv.core.Algorithm
    public void finalize() {
        delete(this.nativeObj);
    }

    public FlannBasedMatcher() {
        super(FlannBasedMatcher_0());
    }
}