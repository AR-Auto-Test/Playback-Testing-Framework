package b.h.b.i.l;

import androidx.constraintlayout.widget.ConstraintLayout;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import java.util.ArrayList;

/* compiled from: BasicMeasure.java */
/* loaded from: classes.dex */
public class b {

    /* renamed from: a  reason: collision with root package name */
    public final ArrayList<b.h.b.i.d> f1884a = new ArrayList<>();

    /* renamed from: b  reason: collision with root package name */
    public a f1885b = new a();

    /* renamed from: c  reason: collision with root package name */
    public b.h.b.i.e f1886c;

    /* compiled from: BasicMeasure.java */
    /* loaded from: classes.dex */
    public static class a {

        /* renamed from: a  reason: collision with root package name */
        public int f1887a;

        /* renamed from: b  reason: collision with root package name */
        public int f1888b;

        /* renamed from: c  reason: collision with root package name */
        public int f1889c;

        /* renamed from: d  reason: collision with root package name */
        public int f1890d;

        /* renamed from: e  reason: collision with root package name */
        public int f1891e;

        /* renamed from: f  reason: collision with root package name */
        public int f1892f;

        /* renamed from: g  reason: collision with root package name */
        public int f1893g;

        /* renamed from: h  reason: collision with root package name */
        public boolean f1894h;
        public boolean i;
        public int j;
    }

    /* compiled from: BasicMeasure.java */
    /* renamed from: b.h.b.i.l.b$b  reason: collision with other inner class name */
    /* loaded from: classes.dex */
    public interface InterfaceC0029b {
    }

    public b(b.h.b.i.e eVar) {
        this.f1886c = eVar;
    }

    public final boolean a(InterfaceC0029b interfaceC0029b, b.h.b.i.d dVar, int i) {
        this.f1885b.f1887a = dVar.m();
        this.f1885b.f1888b = dVar.q();
        this.f1885b.f1889c = dVar.r();
        this.f1885b.f1890d = dVar.l();
        a aVar = this.f1885b;
        aVar.i = false;
        aVar.j = i;
        boolean z = aVar.f1887a == 3;
        boolean z2 = aVar.f1888b == 3;
        boolean z3 = z && dVar.S > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        boolean z4 = z2 && dVar.S > StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        if (z3 && dVar.n[0] == 4) {
            aVar.f1887a = 1;
        }
        if (z4 && dVar.n[1] == 4) {
            aVar.f1888b = 1;
        }
        ((ConstraintLayout.b) interfaceC0029b).b(dVar, aVar);
        dVar.M(this.f1885b.f1891e);
        dVar.H(this.f1885b.f1892f);
        a aVar2 = this.f1885b;
        dVar.y = aVar2.f1894h;
        dVar.E(aVar2.f1893g);
        a aVar3 = this.f1885b;
        aVar3.j = 0;
        return aVar3.i;
    }

    public final void b(b.h.b.i.e eVar, int i, int i2) {
        int i3 = eVar.X;
        int i4 = eVar.Y;
        eVar.K(0);
        eVar.J(0);
        eVar.Q = i;
        int i5 = eVar.X;
        if (i < i5) {
            eVar.Q = i5;
        }
        eVar.R = i2;
        int i6 = eVar.Y;
        if (i2 < i6) {
            eVar.R = i6;
        }
        eVar.K(i3);
        eVar.J(i4);
        this.f1886c.P();
    }

    public void c(b.h.b.i.e eVar) {
        this.f1884a.clear();
        int size = eVar.l0.size();
        for (int i = 0; i < size; i++) {
            b.h.b.i.d dVar = eVar.l0.get(i);
            if (dVar.m() == 3 || dVar.q() == 3) {
                this.f1884a.add(dVar);
            }
        }
        eVar.W();
    }
}