package g;

/* compiled from: SegmentPool.java */
/* loaded from: classes2.dex */
public final class u {

    /* renamed from: a  reason: collision with root package name */
    public static t f6216a;

    /* renamed from: b  reason: collision with root package name */
    public static long f6217b;

    public static void a(t tVar) {
        if (tVar.f6214f == null && tVar.f6215g == null) {
            if (tVar.f6212d) {
                return;
            }
            synchronized (u.class) {
                long j = f6217b + 8192;
                if (j > 65536) {
                    return;
                }
                f6217b = j;
                tVar.f6214f = f6216a;
                tVar.f6211c = 0;
                tVar.f6210b = 0;
                f6216a = tVar;
                return;
            }
        }
        throw new IllegalArgumentException();
    }

    public static t b() {
        synchronized (u.class) {
            t tVar = f6216a;
            if (tVar != null) {
                f6216a = tVar.f6214f;
                tVar.f6214f = null;
                f6217b -= 8192;
                return tVar;
            }
            return new t();
        }
    }
}