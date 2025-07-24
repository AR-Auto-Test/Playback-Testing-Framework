package pl.droidsonroids.gif;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.os.Build;
import android.system.Os;
import b.v.u.c;
import h.a.b.f;
import java.io.FileDescriptor;
import java.io.IOException;

/* loaded from: classes2.dex */
public final class GifInfoHandle {

    /* renamed from: a  reason: collision with root package name */
    public static final /* synthetic */ int f6267a = 0;

    /* renamed from: b  reason: collision with root package name */
    public volatile long f6268b;

    static {
        try {
            System.loadLibrary("pl_droidsonroids_gif");
        } catch (UnsatisfiedLinkError unused) {
            if (c.f2697a == null) {
                try {
                    c.f2697a = (Context) Class.forName("android.app.ActivityThread").getDeclaredMethod("currentApplication", new Class[0]).invoke(null, new Object[0]);
                } catch (Exception e2) {
                    throw new IllegalStateException("LibraryLoader not initialized. Call LibraryLoader.initialize() before using library classes.", e2);
                }
            }
            new f().c(c.f2697a, "pl_droidsonroids_gif", null, null);
        }
    }

    public GifInfoHandle(String str) {
        this.f6268b = openFile(str);
    }

    public static native int createTempNativeFileDescriptor();

    public static native int extractNativeFileDescriptor(FileDescriptor fileDescriptor, boolean z);

    public static native void free(long j);

    public static native int getCurrentFrameIndex(long j);

    public static native int getCurrentLoop(long j);

    public static native int getCurrentPosition(long j);

    public static native int getDuration(long j);

    public static native int getHeight(long j);

    public static native int getLoopCount(long j);

    public static native int getNativeErrorCode(long j);

    public static native int getNumberOfFrames(long j);

    public static native long[] getSavedState(long j);

    public static native int getWidth(long j);

    public static native boolean isOpaque(long j);

    public static native long openFile(String str);

    public static native long openNativeFileDescriptor(int i, long j);

    public static native long renderFrame(long j, Bitmap bitmap);

    public static native boolean reset(long j);

    public static native long restoreRemainder(long j);

    public static native int restoreSavedState(long j, long[] jArr, Bitmap bitmap);

    public static native void saveRemainder(long j);

    public static native void seekToTime(long j, int i, Bitmap bitmap);

    public static native void setLoopCount(long j, char c2);

    public synchronized int a() {
        return getHeight(this.f6268b);
    }

    public synchronized int b() {
        return getNumberOfFrames(this.f6268b);
    }

    public synchronized int c() {
        return getWidth(this.f6268b);
    }

    public synchronized boolean d() {
        return this.f6268b == 0;
    }

    public void finalize() {
        try {
            synchronized (this) {
                free(this.f6268b);
                this.f6268b = 0L;
            }
        } finally {
            super.finalize();
        }
    }

    public GifInfoHandle(AssetFileDescriptor assetFileDescriptor) {
        int extractNativeFileDescriptor;
        try {
            FileDescriptor fileDescriptor = assetFileDescriptor.getFileDescriptor();
            long startOffset = assetFileDescriptor.getStartOffset();
            if (Build.VERSION.SDK_INT > 27) {
                extractNativeFileDescriptor = createTempNativeFileDescriptor();
                Os.dup2(fileDescriptor, extractNativeFileDescriptor);
            } else {
                extractNativeFileDescriptor = extractNativeFileDescriptor(fileDescriptor, false);
            }
            this.f6268b = openNativeFileDescriptor(extractNativeFileDescriptor, startOffset);
            try {
                assetFileDescriptor.close();
            } catch (IOException unused) {
            }
        } catch (Throwable th) {
            try {
                assetFileDescriptor.close();
            } catch (IOException unused2) {
            }
            throw th;
        }
    }
}