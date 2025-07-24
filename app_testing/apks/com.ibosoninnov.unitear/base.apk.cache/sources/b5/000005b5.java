package b.w.b;

import android.animation.TimeInterpolator;
import android.animation.ValueAnimator;
import android.view.View;
import androidx.recyclerview.widget.RecyclerView;
import com.google.android.material.internal.StaticLayoutBuilderCompat;
import java.util.ArrayList;
import java.util.List;

/* compiled from: DefaultItemAnimator.java */
/* loaded from: classes.dex */
public class k extends w {

    /* renamed from: h  reason: collision with root package name */
    public static TimeInterpolator f2745h;
    public ArrayList<RecyclerView.d0> i = new ArrayList<>();
    public ArrayList<RecyclerView.d0> j = new ArrayList<>();
    public ArrayList<b> k = new ArrayList<>();
    public ArrayList<a> l = new ArrayList<>();
    public ArrayList<ArrayList<RecyclerView.d0>> m = new ArrayList<>();
    public ArrayList<ArrayList<b>> n = new ArrayList<>();
    public ArrayList<ArrayList<a>> o = new ArrayList<>();
    public ArrayList<RecyclerView.d0> p = new ArrayList<>();
    public ArrayList<RecyclerView.d0> q = new ArrayList<>();
    public ArrayList<RecyclerView.d0> r = new ArrayList<>();
    public ArrayList<RecyclerView.d0> s = new ArrayList<>();

    /* compiled from: DefaultItemAnimator.java */
    /* loaded from: classes.dex */
    public static class a {

        /* renamed from: a  reason: collision with root package name */
        public RecyclerView.d0 f2746a;

        /* renamed from: b  reason: collision with root package name */
        public RecyclerView.d0 f2747b;

        /* renamed from: c  reason: collision with root package name */
        public int f2748c;

        /* renamed from: d  reason: collision with root package name */
        public int f2749d;

        /* renamed from: e  reason: collision with root package name */
        public int f2750e;

        /* renamed from: f  reason: collision with root package name */
        public int f2751f;

        public a(RecyclerView.d0 d0Var, RecyclerView.d0 d0Var2, int i, int i2, int i3, int i4) {
            this.f2746a = d0Var;
            this.f2747b = d0Var2;
            this.f2748c = i;
            this.f2749d = i2;
            this.f2750e = i3;
            this.f2751f = i4;
        }

        public String toString() {
            StringBuilder x = c.b.a.a.a.x("ChangeInfo{oldHolder=");
            x.append(this.f2746a);
            x.append(", newHolder=");
            x.append(this.f2747b);
            x.append(", fromX=");
            x.append(this.f2748c);
            x.append(", fromY=");
            x.append(this.f2749d);
            x.append(", toX=");
            x.append(this.f2750e);
            x.append(", toY=");
            x.append(this.f2751f);
            x.append('}');
            return x.toString();
        }
    }

    /* compiled from: DefaultItemAnimator.java */
    /* loaded from: classes.dex */
    public static class b {

        /* renamed from: a  reason: collision with root package name */
        public RecyclerView.d0 f2752a;

        /* renamed from: b  reason: collision with root package name */
        public int f2753b;

        /* renamed from: c  reason: collision with root package name */
        public int f2754c;

        /* renamed from: d  reason: collision with root package name */
        public int f2755d;

        /* renamed from: e  reason: collision with root package name */
        public int f2756e;

        public b(RecyclerView.d0 d0Var, int i, int i2, int i3, int i4) {
            this.f2752a = d0Var;
            this.f2753b = i;
            this.f2754c = i2;
            this.f2755d = i3;
            this.f2756e = i4;
        }
    }

    @Override // androidx.recyclerview.widget.RecyclerView.l
    public void e(RecyclerView.d0 d0Var) {
        View view = d0Var.itemView;
        view.animate().cancel();
        int size = this.k.size();
        while (true) {
            size--;
            if (size < 0) {
                break;
            } else if (this.k.get(size).f2752a == d0Var) {
                view.setTranslationY(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
                view.setTranslationX(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
                c(d0Var);
                this.k.remove(size);
            }
        }
        l(this.l, d0Var);
        if (this.i.remove(d0Var)) {
            view.setAlpha(1.0f);
            c(d0Var);
        }
        if (this.j.remove(d0Var)) {
            view.setAlpha(1.0f);
            c(d0Var);
        }
        for (int size2 = this.o.size() - 1; size2 >= 0; size2--) {
            ArrayList<a> arrayList = this.o.get(size2);
            l(arrayList, d0Var);
            if (arrayList.isEmpty()) {
                this.o.remove(size2);
            }
        }
        for (int size3 = this.n.size() - 1; size3 >= 0; size3--) {
            ArrayList<b> arrayList2 = this.n.get(size3);
            int size4 = arrayList2.size() - 1;
            while (true) {
                if (size4 < 0) {
                    break;
                } else if (arrayList2.get(size4).f2752a == d0Var) {
                    view.setTranslationY(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
                    view.setTranslationX(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
                    c(d0Var);
                    arrayList2.remove(size4);
                    if (arrayList2.isEmpty()) {
                        this.n.remove(size3);
                    }
                } else {
                    size4--;
                }
            }
        }
        for (int size5 = this.m.size() - 1; size5 >= 0; size5--) {
            ArrayList<RecyclerView.d0> arrayList3 = this.m.get(size5);
            if (arrayList3.remove(d0Var)) {
                view.setAlpha(1.0f);
                c(d0Var);
                if (arrayList3.isEmpty()) {
                    this.m.remove(size5);
                }
            }
        }
        this.r.remove(d0Var);
        this.p.remove(d0Var);
        this.s.remove(d0Var);
        this.q.remove(d0Var);
        k();
    }

    @Override // androidx.recyclerview.widget.RecyclerView.l
    public void f() {
        int size = this.k.size();
        while (true) {
            size--;
            if (size < 0) {
                break;
            }
            b bVar = this.k.get(size);
            View view = bVar.f2752a.itemView;
            view.setTranslationY(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
            view.setTranslationX(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
            c(bVar.f2752a);
            this.k.remove(size);
        }
        int size2 = this.i.size();
        while (true) {
            size2--;
            if (size2 < 0) {
                break;
            }
            c(this.i.get(size2));
            this.i.remove(size2);
        }
        int size3 = this.j.size();
        while (true) {
            size3--;
            if (size3 < 0) {
                break;
            }
            RecyclerView.d0 d0Var = this.j.get(size3);
            d0Var.itemView.setAlpha(1.0f);
            c(d0Var);
            this.j.remove(size3);
        }
        int size4 = this.l.size();
        while (true) {
            size4--;
            if (size4 < 0) {
                break;
            }
            a aVar = this.l.get(size4);
            RecyclerView.d0 d0Var2 = aVar.f2746a;
            if (d0Var2 != null) {
                m(aVar, d0Var2);
            }
            RecyclerView.d0 d0Var3 = aVar.f2747b;
            if (d0Var3 != null) {
                m(aVar, d0Var3);
            }
        }
        this.l.clear();
        if (!g()) {
            return;
        }
        int size5 = this.n.size();
        while (true) {
            size5--;
            if (size5 < 0) {
                break;
            }
            ArrayList<b> arrayList = this.n.get(size5);
            int size6 = arrayList.size();
            while (true) {
                size6--;
                if (size6 >= 0) {
                    b bVar2 = arrayList.get(size6);
                    View view2 = bVar2.f2752a.itemView;
                    view2.setTranslationY(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
                    view2.setTranslationX(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
                    c(bVar2.f2752a);
                    arrayList.remove(size6);
                    if (arrayList.isEmpty()) {
                        this.n.remove(arrayList);
                    }
                }
            }
        }
        int size7 = this.m.size();
        while (true) {
            size7--;
            if (size7 < 0) {
                break;
            }
            ArrayList<RecyclerView.d0> arrayList2 = this.m.get(size7);
            int size8 = arrayList2.size();
            while (true) {
                size8--;
                if (size8 >= 0) {
                    RecyclerView.d0 d0Var4 = arrayList2.get(size8);
                    d0Var4.itemView.setAlpha(1.0f);
                    c(d0Var4);
                    arrayList2.remove(size8);
                    if (arrayList2.isEmpty()) {
                        this.m.remove(arrayList2);
                    }
                }
            }
        }
        int size9 = this.o.size();
        while (true) {
            size9--;
            if (size9 >= 0) {
                ArrayList<a> arrayList3 = this.o.get(size9);
                int size10 = arrayList3.size();
                while (true) {
                    size10--;
                    if (size10 >= 0) {
                        a aVar2 = arrayList3.get(size10);
                        RecyclerView.d0 d0Var5 = aVar2.f2746a;
                        if (d0Var5 != null) {
                            m(aVar2, d0Var5);
                        }
                        RecyclerView.d0 d0Var6 = aVar2.f2747b;
                        if (d0Var6 != null) {
                            m(aVar2, d0Var6);
                        }
                        if (arrayList3.isEmpty()) {
                            this.o.remove(arrayList3);
                        }
                    }
                }
            } else {
                j(this.r);
                j(this.q);
                j(this.p);
                j(this.s);
                d();
                return;
            }
        }
    }

    @Override // androidx.recyclerview.widget.RecyclerView.l
    public boolean g() {
        return (this.j.isEmpty() && this.l.isEmpty() && this.k.isEmpty() && this.i.isEmpty() && this.q.isEmpty() && this.r.isEmpty() && this.p.isEmpty() && this.s.isEmpty() && this.n.isEmpty() && this.m.isEmpty() && this.o.isEmpty()) ? false : true;
    }

    @Override // b.w.b.w
    public boolean i(RecyclerView.d0 d0Var, int i, int i2, int i3, int i4) {
        View view = d0Var.itemView;
        int translationX = i + ((int) view.getTranslationX());
        int translationY = i2 + ((int) d0Var.itemView.getTranslationY());
        n(d0Var);
        int i5 = i3 - translationX;
        int i6 = i4 - translationY;
        if (i5 == 0 && i6 == 0) {
            c(d0Var);
            return false;
        }
        if (i5 != 0) {
            view.setTranslationX(-i5);
        }
        if (i6 != 0) {
            view.setTranslationY(-i6);
        }
        this.k.add(new b(d0Var, translationX, translationY, i3, i4));
        return true;
    }

    public void j(List<RecyclerView.d0> list) {
        int size = list.size();
        while (true) {
            size--;
            if (size < 0) {
                return;
            }
            list.get(size).itemView.animate().cancel();
        }
    }

    public void k() {
        if (g()) {
            return;
        }
        d();
    }

    public final void l(List<a> list, RecyclerView.d0 d0Var) {
        int size = list.size();
        while (true) {
            size--;
            if (size < 0) {
                return;
            }
            a aVar = list.get(size);
            if (m(aVar, d0Var) && aVar.f2746a == null && aVar.f2747b == null) {
                list.remove(aVar);
            }
        }
    }

    public final boolean m(a aVar, RecyclerView.d0 d0Var) {
        if (aVar.f2747b == d0Var) {
            aVar.f2747b = null;
        } else if (aVar.f2746a != d0Var) {
            return false;
        } else {
            aVar.f2746a = null;
        }
        d0Var.itemView.setAlpha(1.0f);
        d0Var.itemView.setTranslationX(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
        d0Var.itemView.setTranslationY(StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD);
        c(d0Var);
        return true;
    }

    public final void n(RecyclerView.d0 d0Var) {
        if (f2745h == null) {
            f2745h = new ValueAnimator().getInterpolator();
        }
        d0Var.itemView.animate().setInterpolator(f2745h);
        e(d0Var);
    }
}