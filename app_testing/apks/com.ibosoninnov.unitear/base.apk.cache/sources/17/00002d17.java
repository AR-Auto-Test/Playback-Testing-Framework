package f.g0.i;

/* compiled from: Header.java */
/* loaded from: classes2.dex */
public final class c {

    /* renamed from: a  reason: collision with root package name */
    public static final g.h f5873a = g.h.e(":");

    /* renamed from: b  reason: collision with root package name */
    public static final g.h f5874b = g.h.e(":status");

    /* renamed from: c  reason: collision with root package name */
    public static final g.h f5875c = g.h.e(":method");

    /* renamed from: d  reason: collision with root package name */
    public static final g.h f5876d = g.h.e(":path");

    /* renamed from: e  reason: collision with root package name */
    public static final g.h f5877e = g.h.e(":scheme");

    /* renamed from: f  reason: collision with root package name */
    public static final g.h f5878f = g.h.e(":authority");

    /* renamed from: g  reason: collision with root package name */
    public final g.h f5879g;

    /* renamed from: h  reason: collision with root package name */
    public final g.h f5880h;
    public final int i;

    public c(String str, String str2) {
        this(g.h.e(str), g.h.e(str2));
    }

    public boolean equals(Object obj) {
        if (obj instanceof c) {
            c cVar = (c) obj;
            return this.f5879g.equals(cVar.f5879g) && this.f5880h.equals(cVar.f5880h);
        }
        return false;
    }

    public int hashCode() {
        return this.f5880h.hashCode() + ((this.f5879g.hashCode() + 527) * 31);
    }

    public String toString() {
        return f.g0.c.n("%s: %s", this.f5879g.p(), this.f5880h.p());
    }

    public c(g.h hVar, String str) {
        this(hVar, g.h.e(str));
    }

    public c(g.h hVar, g.h hVar2) {
        this.f5879g = hVar;
        this.f5880h = hVar2;
        this.i = hVar2.l() + hVar.l() + 32;
    }
}