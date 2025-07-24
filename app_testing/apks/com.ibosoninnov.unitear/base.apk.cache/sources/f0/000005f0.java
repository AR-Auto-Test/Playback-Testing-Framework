package b.z;

import android.annotation.SuppressLint;
import android.graphics.Rect;
import android.view.View;
import android.view.ViewGroup;
import androidx.fragment.app.Fragment;
import b.j.f.b;
import b.q.b.k0;
import b.z.j;
import java.util.ArrayList;
import java.util.List;

/* compiled from: FragmentTransitionSupport.java */
@SuppressLint({"RestrictedApi"})
/* loaded from: classes.dex */
public class d extends k0 {

    /* compiled from: FragmentTransitionSupport.java */
    /* loaded from: classes.dex */
    public class a extends j.e {

        /* renamed from: a  reason: collision with root package name */
        public final /* synthetic */ Rect f2871a;

        public a(d dVar, Rect rect) {
            this.f2871a = rect;
        }

        @Override // b.z.j.e
        public Rect a(j jVar) {
            return this.f2871a;
        }
    }

    /* compiled from: FragmentTransitionSupport.java */
    /* loaded from: classes.dex */
    public class b implements j.f {

        /* renamed from: a  reason: collision with root package name */
        public final /* synthetic */ View f2872a;

        /* renamed from: b  reason: collision with root package name */
        public final /* synthetic */ ArrayList f2873b;

        public b(d dVar, View view, ArrayList arrayList) {
            this.f2872a = view;
            this.f2873b = arrayList;
        }

        @Override // b.z.j.f
        public void onTransitionCancel(j jVar) {
        }

        @Override // b.z.j.f
        public void onTransitionEnd(j jVar) {
            jVar.removeListener(this);
            this.f2872a.setVisibility(8);
            int size = this.f2873b.size();
            for (int i = 0; i < size; i++) {
                ((View) this.f2873b.get(i)).setVisibility(0);
            }
        }

        @Override // b.z.j.f
        public void onTransitionPause(j jVar) {
        }

        @Override // b.z.j.f
        public void onTransitionResume(j jVar) {
        }

        @Override // b.z.j.f
        public void onTransitionStart(j jVar) {
            jVar.removeListener(this);
            jVar.addListener(this);
        }
    }

    /* compiled from: FragmentTransitionSupport.java */
    /* loaded from: classes.dex */
    public class c extends k {

        /* renamed from: a  reason: collision with root package name */
        public final /* synthetic */ Object f2874a;

        /* renamed from: b  reason: collision with root package name */
        public final /* synthetic */ ArrayList f2875b;

        /* renamed from: c  reason: collision with root package name */
        public final /* synthetic */ Object f2876c;

        /* renamed from: d  reason: collision with root package name */
        public final /* synthetic */ ArrayList f2877d;

        /* renamed from: e  reason: collision with root package name */
        public final /* synthetic */ Object f2878e;

        /* renamed from: f  reason: collision with root package name */
        public final /* synthetic */ ArrayList f2879f;

        public c(Object obj, ArrayList arrayList, Object obj2, ArrayList arrayList2, Object obj3, ArrayList arrayList3) {
            this.f2874a = obj;
            this.f2875b = arrayList;
            this.f2876c = obj2;
            this.f2877d = arrayList2;
            this.f2878e = obj3;
            this.f2879f = arrayList3;
        }

        @Override // b.z.j.f
        public void onTransitionEnd(j jVar) {
            jVar.removeListener(this);
        }

        @Override // b.z.k, b.z.j.f
        public void onTransitionStart(j jVar) {
            Object obj = this.f2874a;
            if (obj != null) {
                d.this.o(obj, this.f2875b, null);
            }
            Object obj2 = this.f2876c;
            if (obj2 != null) {
                d.this.o(obj2, this.f2877d, null);
            }
            Object obj3 = this.f2878e;
            if (obj3 != null) {
                d.this.o(obj3, this.f2879f, null);
            }
        }
    }

    /* compiled from: FragmentTransitionSupport.java */
    /* renamed from: b.z.d$d  reason: collision with other inner class name */
    /* loaded from: classes.dex */
    public class C0057d implements b.a {

        /* renamed from: a  reason: collision with root package name */
        public final /* synthetic */ j f2881a;

        public C0057d(d dVar, j jVar) {
            this.f2881a = jVar;
        }

        @Override // b.j.f.b.a
        public void a() {
            this.f2881a.cancel();
        }
    }

    /* compiled from: FragmentTransitionSupport.java */
    /* loaded from: classes.dex */
    public class e implements j.f {

        /* renamed from: a  reason: collision with root package name */
        public final /* synthetic */ Runnable f2882a;

        public e(d dVar, Runnable runnable) {
            this.f2882a = runnable;
        }

        @Override // b.z.j.f
        public void onTransitionCancel(j jVar) {
        }

        @Override // b.z.j.f
        public void onTransitionEnd(j jVar) {
            this.f2882a.run();
        }

        @Override // b.z.j.f
        public void onTransitionPause(j jVar) {
        }

        @Override // b.z.j.f
        public void onTransitionResume(j jVar) {
        }

        @Override // b.z.j.f
        public void onTransitionStart(j jVar) {
        }
    }

    /* compiled from: FragmentTransitionSupport.java */
    /* loaded from: classes.dex */
    public class f extends j.e {

        /* renamed from: a  reason: collision with root package name */
        public final /* synthetic */ Rect f2883a;

        public f(d dVar, Rect rect) {
            this.f2883a = rect;
        }

        @Override // b.z.j.e
        public Rect a(j jVar) {
            Rect rect = this.f2883a;
            if (rect == null || rect.isEmpty()) {
                return null;
            }
            return this.f2883a;
        }
    }

    public static boolean x(j jVar) {
        return (k0.k(jVar.getTargetIds()) && k0.k(jVar.getTargetNames()) && k0.k(jVar.getTargetTypes())) ? false : true;
    }

    @Override // b.q.b.k0
    public void a(Object obj, View view) {
        if (obj != null) {
            ((j) obj).addTarget(view);
        }
    }

    @Override // b.q.b.k0
    public void b(Object obj, ArrayList<View> arrayList) {
        j jVar = (j) obj;
        if (jVar == null) {
            return;
        }
        int i = 0;
        if (jVar instanceof n) {
            n nVar = (n) jVar;
            int size = nVar.f2903b.size();
            while (i < size) {
                b(nVar.b(i), arrayList);
                i++;
            }
        } else if (x(jVar) || !k0.k(jVar.getTargets())) {
        } else {
            int size2 = arrayList.size();
            while (i < size2) {
                jVar.addTarget(arrayList.get(i));
                i++;
            }
        }
    }

    @Override // b.q.b.k0
    public void c(ViewGroup viewGroup, Object obj) {
        l.a(viewGroup, (j) obj);
    }

    @Override // b.q.b.k0
    public boolean e(Object obj) {
        return obj instanceof j;
    }

    @Override // b.q.b.k0
    public Object g(Object obj) {
        if (obj != null) {
            return ((j) obj).mo0clone();
        }
        return null;
    }

    @Override // b.q.b.k0
    public Object l(Object obj, Object obj2, Object obj3) {
        j jVar = (j) obj;
        j jVar2 = (j) obj2;
        j jVar3 = (j) obj3;
        if (jVar != null && jVar2 != null) {
            n nVar = new n();
            nVar.a(jVar);
            nVar.a(jVar2);
            nVar.e(1);
            jVar = nVar;
        } else if (jVar == null) {
            jVar = jVar2 != null ? jVar2 : null;
        }
        if (jVar3 != null) {
            n nVar2 = new n();
            if (jVar != null) {
                nVar2.a(jVar);
            }
            nVar2.a(jVar3);
            return nVar2;
        }
        return jVar;
    }

    @Override // b.q.b.k0
    public Object m(Object obj, Object obj2, Object obj3) {
        n nVar = new n();
        if (obj != null) {
            nVar.a((j) obj);
        }
        if (obj2 != null) {
            nVar.a((j) obj2);
        }
        if (obj3 != null) {
            nVar.a((j) obj3);
        }
        return nVar;
    }

    @Override // b.q.b.k0
    public void n(Object obj, View view) {
        if (obj != null) {
            ((j) obj).removeTarget(view);
        }
    }

    @Override // b.q.b.k0
    public void o(Object obj, ArrayList<View> arrayList, ArrayList<View> arrayList2) {
        j jVar = (j) obj;
        int i = 0;
        if (jVar instanceof n) {
            n nVar = (n) jVar;
            int size = nVar.f2903b.size();
            while (i < size) {
                o(nVar.b(i), arrayList, arrayList2);
                i++;
            }
        } else if (x(jVar)) {
        } else {
            List<View> targets = jVar.getTargets();
            if (targets.size() != arrayList.size() || !targets.containsAll(arrayList)) {
                return;
            }
            int size2 = arrayList2 == null ? 0 : arrayList2.size();
            while (i < size2) {
                jVar.addTarget(arrayList2.get(i));
                i++;
            }
            int size3 = arrayList.size();
            while (true) {
                size3--;
                if (size3 < 0) {
                    return;
                }
                jVar.removeTarget(arrayList.get(size3));
            }
        }
    }

    @Override // b.q.b.k0
    public void p(Object obj, View view, ArrayList<View> arrayList) {
        ((j) obj).addListener(new b(this, view, arrayList));
    }

    @Override // b.q.b.k0
    public void q(Object obj, Object obj2, ArrayList<View> arrayList, Object obj3, ArrayList<View> arrayList2, Object obj4, ArrayList<View> arrayList3) {
        ((j) obj).addListener(new c(obj2, arrayList, obj3, arrayList2, obj4, arrayList3));
    }

    @Override // b.q.b.k0
    public void r(Object obj, Rect rect) {
        if (obj != null) {
            ((j) obj).setEpicenterCallback(new f(this, rect));
        }
    }

    @Override // b.q.b.k0
    public void s(Object obj, View view) {
        if (view != null) {
            Rect rect = new Rect();
            j(view, rect);
            ((j) obj).setEpicenterCallback(new a(this, rect));
        }
    }

    @Override // b.q.b.k0
    public void t(Fragment fragment, Object obj, b.j.f.b bVar, Runnable runnable) {
        j jVar = (j) obj;
        bVar.a(new C0057d(this, jVar));
        jVar.addListener(new e(this, runnable));
    }

    @Override // b.q.b.k0
    public void u(Object obj, View view, ArrayList<View> arrayList) {
        n nVar = (n) obj;
        List<View> targets = nVar.getTargets();
        targets.clear();
        int size = arrayList.size();
        for (int i = 0; i < size; i++) {
            k0.d(targets, arrayList.get(i));
        }
        targets.add(view);
        arrayList.add(view);
        b(nVar, arrayList);
    }

    @Override // b.q.b.k0
    public void v(Object obj, ArrayList<View> arrayList, ArrayList<View> arrayList2) {
        n nVar = (n) obj;
        if (nVar != null) {
            nVar.getTargets().clear();
            nVar.getTargets().addAll(arrayList2);
            o(nVar, arrayList, arrayList2);
        }
    }

    @Override // b.q.b.k0
    public Object w(Object obj) {
        if (obj == null) {
            return null;
        }
        n nVar = new n();
        nVar.a((j) obj);
        return nVar;
    }
}