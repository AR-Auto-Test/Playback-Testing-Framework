package b.b.g;

import android.view.View;
import android.view.animation.Interpolator;
import b.j.j.s;
import b.j.j.t;
import b.j.j.u;
import java.util.ArrayList;
import java.util.Iterator;

/* compiled from: ViewPropertyAnimatorCompatSet.java */
/* loaded from: classes.dex */
public class g {

    /* renamed from: c  reason: collision with root package name */
    public Interpolator f671c;

    /* renamed from: d  reason: collision with root package name */
    public t f672d;

    /* renamed from: e  reason: collision with root package name */
    public boolean f673e;

    /* renamed from: b  reason: collision with root package name */
    public long f670b = -1;

    /* renamed from: f  reason: collision with root package name */
    public final u f674f = new a();

    /* renamed from: a  reason: collision with root package name */
    public final ArrayList<s> f669a = new ArrayList<>();

    /* compiled from: ViewPropertyAnimatorCompatSet.java */
    /* loaded from: classes.dex */
    public class a extends u {

        /* renamed from: a  reason: collision with root package name */
        public boolean f675a = false;

        /* renamed from: b  reason: collision with root package name */
        public int f676b = 0;

        public a() {
        }

        @Override // b.j.j.t
        public void b(View view) {
            int i = this.f676b + 1;
            this.f676b = i;
            if (i == g.this.f669a.size()) {
                t tVar = g.this.f672d;
                if (tVar != null) {
                    tVar.b(null);
                }
                this.f676b = 0;
                this.f675a = false;
                g.this.f673e = false;
            }
        }

        @Override // b.j.j.u, b.j.j.t
        public void c(View view) {
            if (this.f675a) {
                return;
            }
            this.f675a = true;
            t tVar = g.this.f672d;
            if (tVar != null) {
                tVar.c(null);
            }
        }
    }

    public void a() {
        if (this.f673e) {
            Iterator<s> it = this.f669a.iterator();
            while (it.hasNext()) {
                it.next().b();
            }
            this.f673e = false;
        }
    }

    public void b() {
        View view;
        if (this.f673e) {
            return;
        }
        Iterator<s> it = this.f669a.iterator();
        while (it.hasNext()) {
            s next = it.next();
            long j = this.f670b;
            if (j >= 0) {
                next.c(j);
            }
            Interpolator interpolator = this.f671c;
            if (interpolator != null && (view = next.f2231a.get()) != null) {
                view.animate().setInterpolator(interpolator);
            }
            if (this.f672d != null) {
                next.d(this.f674f);
            }
            View view2 = next.f2231a.get();
            if (view2 != null) {
                view2.animate().start();
            }
        }
        this.f673e = true;
    }
}