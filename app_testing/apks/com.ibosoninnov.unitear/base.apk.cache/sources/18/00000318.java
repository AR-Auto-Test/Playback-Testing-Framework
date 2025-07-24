package b.d.b.d1;

import android.content.Context;

/* compiled from: UseCaseConfigFactory.java */
/* loaded from: classes.dex */
public interface j1 {

    /* compiled from: UseCaseConfigFactory.java */
    /* loaded from: classes.dex */
    public enum a {
        IMAGE_CAPTURE,
        PREVIEW,
        IMAGE_ANALYSIS,
        VIDEO_CAPTURE
    }

    /* compiled from: UseCaseConfigFactory.java */
    /* loaded from: classes.dex */
    public interface b {
        j1 a(Context context);
    }

    i0 a(a aVar);
}