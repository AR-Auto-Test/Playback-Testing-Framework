package b.d.b;

import com.google.common.util.concurrent.ListenableFuture;

/* compiled from: CameraControl.java */
/* loaded from: classes.dex */
public interface f0 {

    /* compiled from: CameraControl.java */
    /* loaded from: classes.dex */
    public static final class a extends Exception {
        public a(String str) {
            super(str);
        }
    }

    ListenableFuture<Void> b(boolean z);
}