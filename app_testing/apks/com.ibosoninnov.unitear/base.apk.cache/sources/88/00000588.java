package b.v;

/* compiled from: NavOptions.java */
/* loaded from: classes.dex */
public final class o {

    /* renamed from: a  reason: collision with root package name */
    public boolean f2662a;

    /* renamed from: b  reason: collision with root package name */
    public int f2663b;

    /* renamed from: c  reason: collision with root package name */
    public boolean f2664c;

    /* renamed from: d  reason: collision with root package name */
    public int f2665d;

    /* renamed from: e  reason: collision with root package name */
    public int f2666e;

    /* renamed from: f  reason: collision with root package name */
    public int f2667f;

    /* renamed from: g  reason: collision with root package name */
    public int f2668g;

    public o(boolean z, int i, boolean z2, int i2, int i3, int i4, int i5) {
        this.f2662a = z;
        this.f2663b = i;
        this.f2664c = z2;
        this.f2665d = i2;
        this.f2666e = i3;
        this.f2667f = i4;
        this.f2668g = i5;
    }

    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (obj == null || o.class != obj.getClass()) {
            return false;
        }
        o oVar = (o) obj;
        return this.f2662a == oVar.f2662a && this.f2663b == oVar.f2663b && this.f2664c == oVar.f2664c && this.f2665d == oVar.f2665d && this.f2666e == oVar.f2666e && this.f2667f == oVar.f2667f && this.f2668g == oVar.f2668g;
    }

    public int hashCode() {
        return ((((((((((((this.f2662a ? 1 : 0) * 31) + this.f2663b) * 31) + (this.f2664c ? 1 : 0)) * 31) + this.f2665d) * 31) + this.f2666e) * 31) + this.f2667f) * 31) + this.f2668g;
    }
}