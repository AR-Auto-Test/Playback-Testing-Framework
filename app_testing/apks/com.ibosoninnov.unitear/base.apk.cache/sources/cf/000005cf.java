package b.w.b;

import android.view.View;

/* compiled from: ViewBoundsCheck.java */
/* loaded from: classes.dex */
public class y {

    /* renamed from: a  reason: collision with root package name */
    public final b f2807a;

    /* renamed from: b  reason: collision with root package name */
    public a f2808b = new a();

    /* compiled from: ViewBoundsCheck.java */
    /* loaded from: classes.dex */
    public static class a {

        /* renamed from: a  reason: collision with root package name */
        public int f2809a = 0;

        /* renamed from: b  reason: collision with root package name */
        public int f2810b;

        /* renamed from: c  reason: collision with root package name */
        public int f2811c;

        /* renamed from: d  reason: collision with root package name */
        public int f2812d;

        /* renamed from: e  reason: collision with root package name */
        public int f2813e;

        public boolean a() {
            int i = this.f2809a;
            if ((i & 7) == 0 || (i & (b(this.f2812d, this.f2810b) << 0)) != 0) {
                int i2 = this.f2809a;
                if ((i2 & 112) == 0 || (i2 & (b(this.f2812d, this.f2811c) << 4)) != 0) {
                    int i3 = this.f2809a;
                    if ((i3 & 1792) == 0 || (i3 & (b(this.f2813e, this.f2810b) << 8)) != 0) {
                        int i4 = this.f2809a;
                        return (i4 & 28672) == 0 || (i4 & (b(this.f2813e, this.f2811c) << 12)) != 0;
                    }
                    return false;
                }
                return false;
            }
            return false;
        }

        public int b(int i, int i2) {
            if (i > i2) {
                return 1;
            }
            return i == i2 ? 2 : 4;
        }
    }

    /* compiled from: ViewBoundsCheck.java */
    /* loaded from: classes.dex */
    public interface b {
        int a(View view);

        int b();

        int c();

        View d(int i);

        int e(View view);
    }

    public y(b bVar) {
        this.f2807a = bVar;
    }

    public View a(int i, int i2, int i3, int i4) {
        int b2 = this.f2807a.b();
        int c2 = this.f2807a.c();
        int i5 = i2 > i ? 1 : -1;
        View view = null;
        while (i != i2) {
            View d2 = this.f2807a.d(i);
            int a2 = this.f2807a.a(d2);
            int e2 = this.f2807a.e(d2);
            a aVar = this.f2808b;
            aVar.f2810b = b2;
            aVar.f2811c = c2;
            aVar.f2812d = a2;
            aVar.f2813e = e2;
            if (i3 != 0) {
                aVar.f2809a = 0;
                aVar.f2809a = i3 | 0;
                if (aVar.a()) {
                    return d2;
                }
            }
            if (i4 != 0) {
                a aVar2 = this.f2808b;
                aVar2.f2809a = 0;
                aVar2.f2809a = i4 | 0;
                if (aVar2.a()) {
                    view = d2;
                }
            }
            i += i5;
        }
        return view;
    }

    public boolean b(View view, int i) {
        a aVar = this.f2808b;
        int b2 = this.f2807a.b();
        int c2 = this.f2807a.c();
        int a2 = this.f2807a.a(view);
        int e2 = this.f2807a.e(view);
        aVar.f2810b = b2;
        aVar.f2811c = c2;
        aVar.f2812d = a2;
        aVar.f2813e = e2;
        if (i != 0) {
            a aVar2 = this.f2808b;
            aVar2.f2809a = 0;
            aVar2.f2809a = 0 | i;
            return aVar2.a();
        }
        return false;
    }
}