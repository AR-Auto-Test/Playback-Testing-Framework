package androidx.databinding;

import android.annotation.TargetApi;
import android.os.Build;
import android.os.Handler;
import android.os.Looper;
import android.util.SparseIntArray;
import android.view.Choreographer;
import android.view.View;
import android.view.ViewGroup;
import androidx.lifecycle.LiveData;
import b.m.g;
import b.m.j;
import b.t.e;
import b.t.n;
import b.t.o;
import com.ibosoninnov.unitear.R;
import java.lang.ref.Reference;
import java.lang.ref.ReferenceQueue;
import java.lang.ref.WeakReference;

/* loaded from: classes.dex */
public abstract class ViewDataBinding extends b.m.a {

    /* renamed from: c  reason: collision with root package name */
    public static int f257c;

    /* renamed from: d  reason: collision with root package name */
    public static final boolean f258d;

    /* renamed from: e  reason: collision with root package name */
    public static final d f259e;

    /* renamed from: f  reason: collision with root package name */
    public static final ReferenceQueue<ViewDataBinding> f260f;

    /* renamed from: g  reason: collision with root package name */
    public static final View.OnAttachStateChangeListener f261g;

    /* renamed from: h  reason: collision with root package name */
    public final Runnable f262h;
    public boolean i;
    public boolean j;
    public g[] k;
    public final View l;
    public boolean m;
    public Choreographer n;
    public final Choreographer.FrameCallback o;
    public Handler p;
    public final b.m.e q;

    /* loaded from: classes.dex */
    public static class OnStartListener implements b.t.g {
        @o(e.a.ON_START)
        public void onStart() {
            throw null;
        }
    }

    /* loaded from: classes.dex */
    public static class a implements d {
        @Override // androidx.databinding.ViewDataBinding.d
        public g a(ViewDataBinding viewDataBinding, int i) {
            return new h(viewDataBinding, i).f269a;
        }
    }

    /* loaded from: classes.dex */
    public static class b implements View.OnAttachStateChangeListener {
        @Override // android.view.View.OnAttachStateChangeListener
        @TargetApi(19)
        public void onViewAttachedToWindow(View view) {
            (view != null ? (ViewDataBinding) view.getTag(R.id.dataBinding) : null).f262h.run();
            view.removeOnAttachStateChangeListener(this);
        }

        @Override // android.view.View.OnAttachStateChangeListener
        public void onViewDetachedFromWindow(View view) {
        }
    }

    /* loaded from: classes.dex */
    public class c implements Runnable {
        public c() {
        }

        @Override // java.lang.Runnable
        public void run() {
            synchronized (this) {
                ViewDataBinding.this.i = false;
            }
            while (true) {
                Reference<? extends ViewDataBinding> poll = ViewDataBinding.f260f.poll();
                if (poll == null) {
                    break;
                } else if (poll instanceof g) {
                    ((g) poll).a();
                }
            }
            if (!ViewDataBinding.this.l.isAttachedToWindow()) {
                View view = ViewDataBinding.this.l;
                View.OnAttachStateChangeListener onAttachStateChangeListener = ViewDataBinding.f261g;
                view.removeOnAttachStateChangeListener(onAttachStateChangeListener);
                ViewDataBinding.this.l.addOnAttachStateChangeListener(onAttachStateChangeListener);
                return;
            }
            ViewDataBinding.this.d();
        }
    }

    /* loaded from: classes.dex */
    public interface d {
        g a(ViewDataBinding viewDataBinding, int i);
    }

    /* loaded from: classes.dex */
    public static class e implements n, f<LiveData<?>> {

        /* renamed from: a  reason: collision with root package name */
        public final g<LiveData<?>> f264a;

        /* renamed from: b  reason: collision with root package name */
        public b.t.h f265b;

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
        @Override // androidx.databinding.ViewDataBinding.f
        public void a(LiveData<?> liveData) {
            liveData.g(this);
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
        @Override // androidx.databinding.ViewDataBinding.f
        public void b(LiveData<?> liveData) {
            LiveData<?> liveData2 = liveData;
            b.t.h hVar = this.f265b;
            if (hVar != null) {
                liveData2.d(hVar, this);
            }
        }
    }

    /* loaded from: classes.dex */
    public interface f<T> {
        void a(T t);

        void b(T t);
    }

    /* loaded from: classes.dex */
    public static class g<T> extends WeakReference<ViewDataBinding> {

        /* renamed from: a  reason: collision with root package name */
        public final f<T> f266a;

        /* renamed from: b  reason: collision with root package name */
        public final int f267b;

        /* renamed from: c  reason: collision with root package name */
        public T f268c;

        public g(ViewDataBinding viewDataBinding, int i, f<T> fVar) {
            super(viewDataBinding, ViewDataBinding.f260f);
            this.f267b = i;
            this.f266a = fVar;
        }

        public boolean a() {
            boolean z;
            T t = this.f268c;
            if (t != null) {
                this.f266a.a(t);
                z = true;
            } else {
                z = false;
            }
            this.f268c = null;
            return z;
        }
    }

    /* loaded from: classes.dex */
    public static class h extends g.a implements f<b.m.g> {

        /* renamed from: a  reason: collision with root package name */
        public final g<b.m.g> f269a;

        public h(ViewDataBinding viewDataBinding, int i) {
            this.f269a = new g<>(viewDataBinding, i, this);
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
        @Override // androidx.databinding.ViewDataBinding.f
        public void a(b.m.g gVar) {
            gVar.b(this);
        }

        /* JADX DEBUG: Method arguments types fixed to match base method, original types: [java.lang.Object] */
        @Override // androidx.databinding.ViewDataBinding.f
        public void b(b.m.g gVar) {
            gVar.a(this);
        }

        @Override // b.m.g.a
        public void c(b.m.g gVar, int i) {
            g<b.m.g> gVar2 = this.f269a;
            ViewDataBinding viewDataBinding = (ViewDataBinding) gVar2.get();
            if (viewDataBinding == null) {
                gVar2.a();
            }
            if (viewDataBinding == null) {
                return;
            }
            g<b.m.g> gVar3 = this.f269a;
            if (gVar3.f268c != gVar) {
                return;
            }
            int i2 = gVar3.f267b;
            int i3 = ViewDataBinding.f257c;
            if (viewDataBinding.i(i2, gVar, i)) {
                viewDataBinding.l();
            }
        }
    }

    static {
        int i = Build.VERSION.SDK_INT;
        f257c = i;
        f258d = i >= 16;
        f259e = new a();
        f260f = new ReferenceQueue<>();
        f261g = new b();
    }

    public ViewDataBinding(Object obj, View view, int i) {
        b.m.e eVar;
        if (obj == null) {
            eVar = null;
        } else if (obj instanceof b.m.e) {
            eVar = (b.m.e) obj;
        } else {
            throw new IllegalArgumentException("The provided bindingComponent parameter must be an instance of DataBindingComponent. See  https://issuetracker.google.com/issues/116541301 for details of why this parameter is not defined as DataBindingComponent");
        }
        this.f262h = new c();
        this.i = false;
        this.j = false;
        this.q = eVar;
        this.k = new g[i];
        this.l = view;
        if (Looper.myLooper() != null) {
            if (f258d) {
                this.n = Choreographer.getInstance();
                this.o = new j(this);
                return;
            }
            this.o = null;
            this.p = new Handler(Looper.myLooper());
            return;
        }
        throw new IllegalStateException("DataBinding must be created in view's UI Thread");
    }

    public static boolean f(String str, int i) {
        int length = str.length();
        if (length == i) {
            return false;
        }
        while (i < length) {
            if (!Character.isDigit(str.charAt(i))) {
                return false;
            }
            i++;
        }
        return true;
    }

    public static void g(b.m.e eVar, View view, Object[] objArr, SparseIntArray sparseIntArray, boolean z) {
        int id;
        int i;
        if ((view != null ? (ViewDataBinding) view.getTag(R.id.dataBinding) : null) != null) {
            return;
        }
        Object tag = view.getTag();
        String str = tag instanceof String ? (String) tag : null;
        boolean z2 = true;
        if (z && str != null && str.startsWith("layout")) {
            int lastIndexOf = str.lastIndexOf(95);
            if (lastIndexOf > 0) {
                int i2 = lastIndexOf + 1;
                if (f(str, i2)) {
                    int j = j(str, i2);
                    if (objArr[j] == null) {
                        objArr[j] = view;
                    }
                }
            }
            z2 = false;
        } else {
            if (str != null && str.startsWith("binding_")) {
                int j2 = j(str, 8);
                if (objArr[j2] == null) {
                    objArr[j2] = view;
                }
            }
            z2 = false;
        }
        if (!z2 && (id = view.getId()) > 0 && sparseIntArray != null && (i = sparseIntArray.get(id, -1)) >= 0 && objArr[i] == null) {
            objArr[i] = view;
        }
        if (view instanceof ViewGroup) {
            ViewGroup viewGroup = (ViewGroup) view;
            int childCount = viewGroup.getChildCount();
            for (int i3 = 0; i3 < childCount; i3++) {
                g(eVar, viewGroup.getChildAt(i3), objArr, sparseIntArray, false);
            }
        }
    }

    public static Object[] h(b.m.e eVar, View view, int i, SparseIntArray sparseIntArray) {
        Object[] objArr = new Object[i];
        g(eVar, view, objArr, sparseIntArray, true);
        return objArr;
    }

    public static int j(String str, int i) {
        int length = str.length();
        int i2 = 0;
        while (i < length) {
            i2 = (i2 * 10) + (str.charAt(i) - '0');
            i++;
        }
        return i2;
    }

    public abstract void c();

    public void d() {
        if (this.m) {
            l();
        } else if (e()) {
            this.m = true;
            this.j = false;
            c();
            this.m = false;
        }
    }

    public abstract boolean e();

    public abstract boolean i(int i, Object obj, int i2);

    /* JADX DEBUG: Multi-variable search result rejected for r3v0, resolved type: java.lang.Object */
    /* JADX WARN: Multi-variable type inference failed */
    public void k(int i, Object obj, d dVar) {
        g gVar = this.k[i];
        if (gVar == null) {
            gVar = dVar.a(this, i);
            this.k[i] = gVar;
        }
        gVar.a();
        gVar.f268c = obj;
        gVar.f266a.b(obj);
    }

    public void l() {
        synchronized (this) {
            if (this.i) {
                return;
            }
            this.i = true;
            if (f258d) {
                this.n.postFrameCallback(this.o);
            } else {
                this.p.post(this.f262h);
            }
        }
    }
}