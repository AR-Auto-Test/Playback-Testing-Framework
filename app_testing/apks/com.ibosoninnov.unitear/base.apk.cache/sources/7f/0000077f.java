package c.c.a.m.v;

/* compiled from: DiskCacheStrategy.java */
/* loaded from: classes.dex */
public abstract class k {

    /* renamed from: a  reason: collision with root package name */
    public static final k f3731a = new a();

    /* renamed from: b  reason: collision with root package name */
    public static final k f3732b = new b();

    /* renamed from: c  reason: collision with root package name */
    public static final k f3733c = new c();

    /* compiled from: DiskCacheStrategy.java */
    /* loaded from: classes.dex */
    public class a extends k {
        @Override // c.c.a.m.v.k
        public boolean a() {
            return false;
        }

        @Override // c.c.a.m.v.k
        public boolean b() {
            return false;
        }

        @Override // c.c.a.m.v.k
        public boolean c(c.c.a.m.a aVar) {
            return false;
        }

        @Override // c.c.a.m.v.k
        public boolean d(boolean z, c.c.a.m.a aVar, c.c.a.m.c cVar) {
            return false;
        }
    }

    /* compiled from: DiskCacheStrategy.java */
    /* loaded from: classes.dex */
    public class b extends k {
        @Override // c.c.a.m.v.k
        public boolean a() {
            return true;
        }

        @Override // c.c.a.m.v.k
        public boolean b() {
            return false;
        }

        @Override // c.c.a.m.v.k
        public boolean c(c.c.a.m.a aVar) {
            return (aVar == c.c.a.m.a.DATA_DISK_CACHE || aVar == c.c.a.m.a.MEMORY_CACHE) ? false : true;
        }

        @Override // c.c.a.m.v.k
        public boolean d(boolean z, c.c.a.m.a aVar, c.c.a.m.c cVar) {
            return false;
        }
    }

    /* compiled from: DiskCacheStrategy.java */
    /* loaded from: classes.dex */
    public class c extends k {
        @Override // c.c.a.m.v.k
        public boolean a() {
            return true;
        }

        @Override // c.c.a.m.v.k
        public boolean b() {
            return true;
        }

        @Override // c.c.a.m.v.k
        public boolean c(c.c.a.m.a aVar) {
            return aVar == c.c.a.m.a.REMOTE;
        }

        @Override // c.c.a.m.v.k
        public boolean d(boolean z, c.c.a.m.a aVar, c.c.a.m.c cVar) {
            return ((z && aVar == c.c.a.m.a.DATA_DISK_CACHE) || aVar == c.c.a.m.a.LOCAL) && cVar == c.c.a.m.c.TRANSFORMED;
        }
    }

    public abstract boolean a();

    public abstract boolean b();

    public abstract boolean c(c.c.a.m.a aVar);

    public abstract boolean d(boolean z, c.c.a.m.a aVar, c.c.a.m.c cVar);
}