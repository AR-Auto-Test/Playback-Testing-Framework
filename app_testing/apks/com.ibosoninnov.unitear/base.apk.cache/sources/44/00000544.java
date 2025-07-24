package b.q.b;

import androidx.fragment.app.Fragment;
import b.t.e;
import java.util.ArrayList;

/* compiled from: FragmentTransaction.java */
/* loaded from: classes.dex */
public abstract class y {

    /* renamed from: b  reason: collision with root package name */
    public int f2542b;

    /* renamed from: c  reason: collision with root package name */
    public int f2543c;

    /* renamed from: d  reason: collision with root package name */
    public int f2544d;

    /* renamed from: e  reason: collision with root package name */
    public int f2545e;

    /* renamed from: f  reason: collision with root package name */
    public int f2546f;

    /* renamed from: g  reason: collision with root package name */
    public boolean f2547g;
    public String i;
    public int j;
    public CharSequence k;
    public int l;
    public CharSequence m;
    public ArrayList<String> n;
    public ArrayList<String> o;

    /* renamed from: a  reason: collision with root package name */
    public ArrayList<a> f2541a = new ArrayList<>();

    /* renamed from: h  reason: collision with root package name */
    public boolean f2548h = true;
    public boolean p = false;

    /* compiled from: FragmentTransaction.java */
    /* loaded from: classes.dex */
    public static final class a {

        /* renamed from: a  reason: collision with root package name */
        public int f2549a;

        /* renamed from: b  reason: collision with root package name */
        public Fragment f2550b;

        /* renamed from: c  reason: collision with root package name */
        public int f2551c;

        /* renamed from: d  reason: collision with root package name */
        public int f2552d;

        /* renamed from: e  reason: collision with root package name */
        public int f2553e;

        /* renamed from: f  reason: collision with root package name */
        public int f2554f;

        /* renamed from: g  reason: collision with root package name */
        public e.b f2555g;

        /* renamed from: h  reason: collision with root package name */
        public e.b f2556h;

        public a() {
        }

        public a(int i, Fragment fragment) {
            this.f2549a = i;
            this.f2550b = fragment;
            e.b bVar = e.b.RESUMED;
            this.f2555g = bVar;
            this.f2556h = bVar;
        }
    }

    public y(m mVar, ClassLoader classLoader) {
    }

    public void b(a aVar) {
        this.f2541a.add(aVar);
        aVar.f2551c = this.f2542b;
        aVar.f2552d = this.f2543c;
        aVar.f2553e = this.f2544d;
        aVar.f2554f = this.f2545e;
    }

    public abstract int c();

    public abstract void d(int i, Fragment fragment, String str, int i2);
}