package b.z;

import android.view.View;
import android.view.ViewGroup;
import android.view.ViewTreeObserver;
import com.ibosoninnov.unitear.R;
import java.lang.ref.WeakReference;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.concurrent.atomic.AtomicInteger;

/* compiled from: TransitionManager.java */
/* loaded from: classes.dex */
public class l {

    /* renamed from: a  reason: collision with root package name */
    public static j f2896a = new b.z.a();

    /* renamed from: b  reason: collision with root package name */
    public static ThreadLocal<WeakReference<b.f.a<ViewGroup, ArrayList<j>>>> f2897b = new ThreadLocal<>();

    /* renamed from: c  reason: collision with root package name */
    public static ArrayList<ViewGroup> f2898c = new ArrayList<>();

    /* compiled from: TransitionManager.java */
    /* loaded from: classes.dex */
    public static class a implements ViewTreeObserver.OnPreDrawListener, View.OnAttachStateChangeListener {

        /* renamed from: b  reason: collision with root package name */
        public j f2899b;

        /* renamed from: c  reason: collision with root package name */
        public ViewGroup f2900c;

        /* compiled from: TransitionManager.java */
        /* renamed from: b.z.l$a$a  reason: collision with other inner class name */
        /* loaded from: classes.dex */
        public class C0058a extends k {

            /* renamed from: a  reason: collision with root package name */
            public final /* synthetic */ b.f.a f2901a;

            public C0058a(b.f.a aVar) {
                this.f2901a = aVar;
            }

            @Override // b.z.j.f
            public void onTransitionEnd(j jVar) {
                ((ArrayList) this.f2901a.get(a.this.f2900c)).remove(jVar);
                jVar.removeListener(this);
            }
        }

        public a(j jVar, ViewGroup viewGroup) {
            this.f2899b = jVar;
            this.f2900c = viewGroup;
        }

        @Override // android.view.ViewTreeObserver.OnPreDrawListener
        public boolean onPreDraw() {
            this.f2900c.getViewTreeObserver().removeOnPreDrawListener(this);
            this.f2900c.removeOnAttachStateChangeListener(this);
            if (l.f2898c.remove(this.f2900c)) {
                b.f.a<ViewGroup, ArrayList<j>> b2 = l.b();
                ArrayList<j> arrayList = b2.get(this.f2900c);
                ArrayList arrayList2 = null;
                if (arrayList == null) {
                    arrayList = new ArrayList<>();
                    b2.put(this.f2900c, arrayList);
                } else if (arrayList.size() > 0) {
                    arrayList2 = new ArrayList(arrayList);
                }
                arrayList.add(this.f2899b);
                this.f2899b.addListener(new C0058a(b2));
                this.f2899b.captureValues(this.f2900c, false);
                if (arrayList2 != null) {
                    Iterator it = arrayList2.iterator();
                    while (it.hasNext()) {
                        ((j) it.next()).resume(this.f2900c);
                    }
                }
                this.f2899b.playTransition(this.f2900c);
                return true;
            }
            return true;
        }

        @Override // android.view.View.OnAttachStateChangeListener
        public void onViewAttachedToWindow(View view) {
        }

        @Override // android.view.View.OnAttachStateChangeListener
        public void onViewDetachedFromWindow(View view) {
            this.f2900c.getViewTreeObserver().removeOnPreDrawListener(this);
            this.f2900c.removeOnAttachStateChangeListener(this);
            l.f2898c.remove(this.f2900c);
            ArrayList<j> arrayList = l.b().get(this.f2900c);
            if (arrayList != null && arrayList.size() > 0) {
                Iterator<j> it = arrayList.iterator();
                while (it.hasNext()) {
                    it.next().resume(this.f2900c);
                }
            }
            this.f2899b.clearValues(true);
        }
    }

    public static void a(ViewGroup viewGroup, j jVar) {
        if (f2898c.contains(viewGroup)) {
            return;
        }
        AtomicInteger atomicInteger = b.j.j.q.f2214a;
        if (viewGroup.isLaidOut()) {
            f2898c.add(viewGroup);
            if (jVar == null) {
                jVar = f2896a;
            }
            j mo0clone = jVar.mo0clone();
            ArrayList<j> orDefault = b().getOrDefault(viewGroup, null);
            if (orDefault != null && orDefault.size() > 0) {
                Iterator<j> it = orDefault.iterator();
                while (it.hasNext()) {
                    it.next().pause(viewGroup);
                }
            }
            if (mo0clone != null) {
                mo0clone.captureValues(viewGroup, true);
            }
            if (((h) viewGroup.getTag(R.id.transition_current_scene)) == null) {
                viewGroup.setTag(R.id.transition_current_scene, null);
                if (mo0clone != null) {
                    a aVar = new a(mo0clone, viewGroup);
                    viewGroup.addOnAttachStateChangeListener(aVar);
                    viewGroup.getViewTreeObserver().addOnPreDrawListener(aVar);
                    return;
                }
                return;
            }
            throw null;
        }
    }

    public static b.f.a<ViewGroup, ArrayList<j>> b() {
        b.f.a<ViewGroup, ArrayList<j>> aVar;
        WeakReference<b.f.a<ViewGroup, ArrayList<j>>> weakReference = f2897b.get();
        if (weakReference == null || (aVar = weakReference.get()) == null) {
            b.f.a<ViewGroup, ArrayList<j>> aVar2 = new b.f.a<>();
            f2897b.set(new WeakReference<>(aVar2));
            return aVar2;
        }
        return aVar;
    }
}