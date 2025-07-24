package c.a.a.z.l;

import c.a.a.z.j.j;
import c.a.a.z.j.k;
import c.a.a.z.j.l;
import java.util.List;
import java.util.Locale;

/* compiled from: Layer.java */
/* loaded from: classes.dex */
public class e {

    /* renamed from: a  reason: collision with root package name */
    public final List<c.a.a.z.k.b> f3395a;

    /* renamed from: b  reason: collision with root package name */
    public final c.a.a.d f3396b;

    /* renamed from: c  reason: collision with root package name */
    public final String f3397c;

    /* renamed from: d  reason: collision with root package name */
    public final long f3398d;

    /* renamed from: e  reason: collision with root package name */
    public final a f3399e;

    /* renamed from: f  reason: collision with root package name */
    public final long f3400f;

    /* renamed from: g  reason: collision with root package name */
    public final String f3401g;

    /* renamed from: h  reason: collision with root package name */
    public final List<c.a.a.z.k.f> f3402h;
    public final l i;
    public final int j;
    public final int k;
    public final int l;
    public final float m;
    public final float n;
    public final int o;
    public final int p;
    public final j q;
    public final k r;
    public final c.a.a.z.j.b s;
    public final List<c.a.a.d0.a<Float>> t;
    public final int u;
    public final boolean v;

    /* compiled from: Layer.java */
    /* loaded from: classes.dex */
    public enum a {
        PRE_COMP,
        SOLID,
        IMAGE,
        NULL,
        SHAPE,
        TEXT,
        UNKNOWN
    }

    /* JADX WARN: Incorrect types in method signature: (Ljava/util/List<Lc/a/a/z/k/b;>;Lc/a/a/d;Ljava/lang/String;JLc/a/a/z/l/e$a;JLjava/lang/String;Ljava/util/List<Lc/a/a/z/k/f;>;Lc/a/a/z/j/l;IIIFFIILc/a/a/z/j/j;Lc/a/a/z/j/k;Ljava/util/List<Lc/a/a/d0/a<Ljava/lang/Float;>;>;Ljava/lang/Object;Lc/a/a/z/j/b;Z)V */
    public e(List list, c.a.a.d dVar, String str, long j, a aVar, long j2, String str2, List list2, l lVar, int i, int i2, int i3, float f2, float f3, int i4, int i5, j jVar, k kVar, List list3, int i6, c.a.a.z.j.b bVar, boolean z) {
        this.f3395a = list;
        this.f3396b = dVar;
        this.f3397c = str;
        this.f3398d = j;
        this.f3399e = aVar;
        this.f3400f = j2;
        this.f3401g = str2;
        this.f3402h = list2;
        this.i = lVar;
        this.j = i;
        this.k = i2;
        this.l = i3;
        this.m = f2;
        this.n = f3;
        this.o = i4;
        this.p = i5;
        this.q = jVar;
        this.r = kVar;
        this.t = list3;
        this.u = i6;
        this.s = bVar;
        this.v = z;
    }

    public String a(String str) {
        StringBuilder x = c.b.a.a.a.x(str);
        x.append(this.f3397c);
        x.append("\n");
        e e2 = this.f3396b.e(this.f3400f);
        if (e2 != null) {
            x.append("\t\tParents: ");
            x.append(e2.f3397c);
            e e3 = this.f3396b.e(e2.f3400f);
            while (e3 != null) {
                x.append("->");
                x.append(e3.f3397c);
                e3 = this.f3396b.e(e3.f3400f);
            }
            x.append(str);
            x.append("\n");
        }
        if (!this.f3402h.isEmpty()) {
            x.append(str);
            x.append("\tMasks: ");
            x.append(this.f3402h.size());
            x.append("\n");
        }
        if (this.j != 0 && this.k != 0) {
            x.append(str);
            x.append("\tBackground: ");
            x.append(String.format(Locale.US, "%dx%d %X\n", Integer.valueOf(this.j), Integer.valueOf(this.k), Integer.valueOf(this.l)));
        }
        if (!this.f3395a.isEmpty()) {
            x.append(str);
            x.append("\tShapes:\n");
            for (c.a.a.z.k.b bVar : this.f3395a) {
                x.append(str);
                x.append("\t\t");
                x.append(bVar);
                x.append("\n");
            }
        }
        return x.toString();
    }

    public String toString() {
        return a("");
    }
}