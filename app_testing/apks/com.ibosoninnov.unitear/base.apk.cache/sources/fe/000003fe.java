package b.h.b.i.l;

import java.util.ArrayList;
import java.util.List;

/* compiled from: DependencyNode.java */
/* loaded from: classes.dex */
public class f implements d {

    /* renamed from: d  reason: collision with root package name */
    public o f1906d;

    /* renamed from: f  reason: collision with root package name */
    public int f1908f;

    /* renamed from: g  reason: collision with root package name */
    public int f1909g;

    /* renamed from: a  reason: collision with root package name */
    public d f1903a = null;

    /* renamed from: b  reason: collision with root package name */
    public boolean f1904b = false;

    /* renamed from: c  reason: collision with root package name */
    public boolean f1905c = false;

    /* renamed from: e  reason: collision with root package name */
    public a f1907e = a.UNKNOWN;

    /* renamed from: h  reason: collision with root package name */
    public int f1910h = 1;
    public g i = null;
    public boolean j = false;
    public List<d> k = new ArrayList();
    public List<f> l = new ArrayList();

    /* compiled from: DependencyNode.java */
    /* loaded from: classes.dex */
    public enum a {
        UNKNOWN,
        HORIZONTAL_DIMENSION,
        VERTICAL_DIMENSION,
        LEFT,
        RIGHT,
        TOP,
        BOTTOM,
        BASELINE
    }

    public f(o oVar) {
        this.f1906d = oVar;
    }

    @Override // b.h.b.i.l.d
    public void a(d dVar) {
        for (f fVar : this.l) {
            if (!fVar.j) {
                return;
            }
        }
        this.f1905c = true;
        d dVar2 = this.f1903a;
        if (dVar2 != null) {
            dVar2.a(this);
        }
        if (this.f1904b) {
            this.f1906d.a(this);
            return;
        }
        f fVar2 = null;
        int i = 0;
        for (f fVar3 : this.l) {
            if (!(fVar3 instanceof g)) {
                i++;
                fVar2 = fVar3;
            }
        }
        if (fVar2 != null && i == 1 && fVar2.j) {
            g gVar = this.i;
            if (gVar != null) {
                if (!gVar.j) {
                    return;
                }
                this.f1908f = this.f1910h * gVar.f1909g;
            }
            c(fVar2.f1909g + this.f1908f);
        }
        d dVar3 = this.f1903a;
        if (dVar3 != null) {
            dVar3.a(this);
        }
    }

    public void b() {
        this.l.clear();
        this.k.clear();
        this.j = false;
        this.f1909g = 0;
        this.f1905c = false;
        this.f1904b = false;
    }

    public void c(int i) {
        if (this.j) {
            return;
        }
        this.j = true;
        this.f1909g = i;
        for (d dVar : this.k) {
            dVar.a(dVar);
        }
    }

    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(this.f1906d.f1929b.d0);
        sb.append(":");
        sb.append(this.f1907e);
        sb.append("(");
        sb.append(this.j ? Integer.valueOf(this.f1909g) : "unresolved");
        sb.append(") <t=");
        sb.append(this.l.size());
        sb.append(":d=");
        sb.append(this.k.size());
        sb.append(">");
        return sb.toString();
    }
}