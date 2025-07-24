package c.c.a.m.x.c;

/* compiled from: DownsampleStrategy.java */
/* loaded from: classes.dex */
public abstract class l {

    /* renamed from: a  reason: collision with root package name */
    public static final l f3969a = new c();

    /* renamed from: b  reason: collision with root package name */
    public static final l f3970b = new a();

    /* renamed from: c  reason: collision with root package name */
    public static final l f3971c;

    /* renamed from: d  reason: collision with root package name */
    public static final l f3972d;

    /* renamed from: e  reason: collision with root package name */
    public static final l f3973e;

    /* renamed from: f  reason: collision with root package name */
    public static final c.c.a.m.o<l> f3974f;

    /* renamed from: g  reason: collision with root package name */
    public static final boolean f3975g;

    /* compiled from: DownsampleStrategy.java */
    /* loaded from: classes.dex */
    public static class a extends l {
        @Override // c.c.a.m.x.c.l
        public int a(int i, int i2, int i3, int i4) {
            return (b(i, i2, i3, i4) == 1.0f || l.f3975g) ? 2 : 1;
        }

        @Override // c.c.a.m.x.c.l
        public float b(int i, int i2, int i3, int i4) {
            return Math.min(1.0f, l.f3969a.b(i, i2, i3, i4));
        }
    }

    /* compiled from: DownsampleStrategy.java */
    /* loaded from: classes.dex */
    public static class b extends l {
        @Override // c.c.a.m.x.c.l
        public int a(int i, int i2, int i3, int i4) {
            return 2;
        }

        @Override // c.c.a.m.x.c.l
        public float b(int i, int i2, int i3, int i4) {
            return Math.max(i3 / i, i4 / i2);
        }
    }

    /* compiled from: DownsampleStrategy.java */
    /* loaded from: classes.dex */
    public static class c extends l {
        @Override // c.c.a.m.x.c.l
        public int a(int i, int i2, int i3, int i4) {
            return l.f3975g ? 2 : 1;
        }

        @Override // c.c.a.m.x.c.l
        public float b(int i, int i2, int i3, int i4) {
            if (l.f3975g) {
                return Math.min(i3 / i, i4 / i2);
            }
            int max = Math.max(i2 / i4, i / i3);
            if (max == 0) {
                return 1.0f;
            }
            return 1.0f / Integer.highestOneBit(max);
        }
    }

    /* compiled from: DownsampleStrategy.java */
    /* loaded from: classes.dex */
    public static class d extends l {
        @Override // c.c.a.m.x.c.l
        public int a(int i, int i2, int i3, int i4) {
            return 2;
        }

        @Override // c.c.a.m.x.c.l
        public float b(int i, int i2, int i3, int i4) {
            return 1.0f;
        }
    }

    static {
        b bVar = new b();
        f3971c = bVar;
        f3972d = new d();
        f3973e = bVar;
        f3974f = c.c.a.m.o.a("com.bumptech.glide.load.resource.bitmap.Downsampler.DownsampleStrategy", bVar);
        f3975g = true;
    }

    public abstract int a(int i, int i2, int i3, int i4);

    public abstract float b(int i, int i2, int i3, int i4);
}