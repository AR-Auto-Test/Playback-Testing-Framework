package c.c.a.s.k;

/* compiled from: StateVerifier.java */
/* loaded from: classes.dex */
public abstract class d {

    /* compiled from: StateVerifier.java */
    /* loaded from: classes.dex */
    public static class b extends d {

        /* renamed from: a  reason: collision with root package name */
        public volatile boolean f4205a;

        public b() {
            super(null);
        }

        @Override // c.c.a.s.k.d
        public void a() {
            if (this.f4205a) {
                throw new IllegalStateException("Already released");
            }
        }
    }

    public d(a aVar) {
    }

    public abstract void a();
}