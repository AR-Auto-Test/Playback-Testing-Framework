package b.j.d;

import android.graphics.Insets;

/* compiled from: Insets.java */
/* loaded from: classes.dex */
public final class b {

    /* renamed from: a  reason: collision with root package name */
    public static final b f2095a = new b(0, 0, 0, 0);

    /* renamed from: b  reason: collision with root package name */
    public final int f2096b;

    /* renamed from: c  reason: collision with root package name */
    public final int f2097c;

    /* renamed from: d  reason: collision with root package name */
    public final int f2098d;

    /* renamed from: e  reason: collision with root package name */
    public final int f2099e;

    public b(int i, int i2, int i3, int i4) {
        this.f2096b = i;
        this.f2097c = i2;
        this.f2098d = i3;
        this.f2099e = i4;
    }

    public static b a(int i, int i2, int i3, int i4) {
        if (i == 0 && i2 == 0 && i3 == 0 && i4 == 0) {
            return f2095a;
        }
        return new b(i, i2, i3, i4);
    }

    public Insets b() {
        return Insets.of(this.f2096b, this.f2097c, this.f2098d, this.f2099e);
    }

    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (obj == null || b.class != obj.getClass()) {
            return false;
        }
        b bVar = (b) obj;
        return this.f2099e == bVar.f2099e && this.f2096b == bVar.f2096b && this.f2098d == bVar.f2098d && this.f2097c == bVar.f2097c;
    }

    public int hashCode() {
        return (((((this.f2096b * 31) + this.f2097c) * 31) + this.f2098d) * 31) + this.f2099e;
    }

    public String toString() {
        StringBuilder x = c.b.a.a.a.x("Insets{left=");
        x.append(this.f2096b);
        x.append(", top=");
        x.append(this.f2097c);
        x.append(", right=");
        x.append(this.f2098d);
        x.append(", bottom=");
        x.append(this.f2099e);
        x.append('}');
        return x.toString();
    }
}