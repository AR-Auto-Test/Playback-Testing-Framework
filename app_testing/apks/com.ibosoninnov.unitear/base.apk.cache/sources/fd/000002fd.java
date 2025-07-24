package b.d.b.d1;

import com.google.auto.value.AutoValue;

/* compiled from: SurfaceConfig.java */
@AutoValue
/* loaded from: classes.dex */
public abstract class e1 {

    /* compiled from: SurfaceConfig.java */
    /* loaded from: classes.dex */
    public enum a {
        ANALYSIS(0),
        PREVIEW(1),
        RECORD(2),
        MAXIMUM(3),
        NOT_SUPPORT(4);
        

        /* renamed from: h  reason: collision with root package name */
        public final int f1451h;

        a(int i) {
            this.f1451h = i;
        }
    }

    /* compiled from: SurfaceConfig.java */
    /* loaded from: classes.dex */
    public enum b {
        PRIV,
        YUV,
        JPEG,
        RAW
    }

    public abstract a a();

    public abstract b b();
}