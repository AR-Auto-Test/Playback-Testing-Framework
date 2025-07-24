package c.c.a.n;

import android.app.Activity;
import android.app.Application;
import android.app.Fragment;
import android.app.FragmentManager;
import android.content.Context;
import android.content.ContextWrapper;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.Message;
import android.util.Log;
import c.c.a.c;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/* compiled from: RequestManagerRetriever.java */
/* loaded from: classes.dex */
public class p implements Handler.Callback {

    /* renamed from: a  reason: collision with root package name */
    public static final b f4091a = new a();

    /* renamed from: b  reason: collision with root package name */
    public volatile c.c.a.i f4092b;

    /* renamed from: c  reason: collision with root package name */
    public final Map<FragmentManager, o> f4093c = new HashMap();

    /* renamed from: d  reason: collision with root package name */
    public final Map<b.q.b.q, s> f4094d = new HashMap();

    /* renamed from: e  reason: collision with root package name */
    public final Handler f4095e;

    /* renamed from: f  reason: collision with root package name */
    public final b f4096f;

    /* renamed from: g  reason: collision with root package name */
    public final k f4097g;

    /* compiled from: RequestManagerRetriever.java */
    /* loaded from: classes.dex */
    public class a implements b {
    }

    /* compiled from: RequestManagerRetriever.java */
    /* loaded from: classes.dex */
    public interface b {
    }

    public p(b bVar, c.c.a.e eVar) {
        k gVar;
        new Bundle();
        this.f4096f = bVar == null ? f4091a : bVar;
        this.f4095e = new Handler(Looper.getMainLooper(), this);
        if (c.c.a.m.x.c.r.f3989b && c.c.a.m.x.c.r.f3988a) {
            if (eVar.f3433a.containsKey(c.d.class)) {
                gVar = new i();
            } else {
                gVar = new j();
            }
        } else {
            gVar = new g();
        }
        this.f4097g = gVar;
    }

    public static Activity a(Context context) {
        if (context instanceof Activity) {
            return (Activity) context;
        }
        if (context instanceof ContextWrapper) {
            return a(((ContextWrapper) context).getBaseContext());
        }
        return null;
    }

    public static boolean g(Context context) {
        Activity a2 = a(context);
        return a2 == null || !a2.isFinishing();
    }

    public c.c.a.i b(Activity activity) {
        if (c.c.a.s.j.h()) {
            return c(activity.getApplicationContext());
        }
        if (activity instanceof b.q.b.d) {
            return d((b.q.b.d) activity);
        }
        if (!activity.isDestroyed()) {
            this.f4097g.a(activity);
            FragmentManager fragmentManager = activity.getFragmentManager();
            boolean g2 = g(activity);
            o e2 = e(fragmentManager, null);
            c.c.a.i iVar = e2.f4087e;
            if (iVar == null) {
                c.c.a.b b2 = c.c.a.b.b(activity);
                b bVar = this.f4096f;
                c.c.a.n.a aVar = e2.f4084b;
                q qVar = e2.f4085c;
                Objects.requireNonNull((a) bVar);
                c.c.a.i iVar2 = new c.c.a.i(b2, aVar, qVar, activity);
                if (g2) {
                    iVar2.onStart();
                }
                e2.f4087e = iVar2;
                return iVar2;
            }
            return iVar;
        }
        throw new IllegalArgumentException("You cannot start a load for a destroyed activity");
    }

    public c.c.a.i c(Context context) {
        if (context != null) {
            if (c.c.a.s.j.i() && !(context instanceof Application)) {
                if (context instanceof b.q.b.d) {
                    return d((b.q.b.d) context);
                }
                if (context instanceof Activity) {
                    return b((Activity) context);
                }
                if (context instanceof ContextWrapper) {
                    ContextWrapper contextWrapper = (ContextWrapper) context;
                    if (contextWrapper.getBaseContext().getApplicationContext() != null) {
                        return c(contextWrapper.getBaseContext());
                    }
                }
            }
            if (this.f4092b == null) {
                synchronized (this) {
                    if (this.f4092b == null) {
                        c.c.a.b b2 = c.c.a.b.b(context.getApplicationContext());
                        b bVar = this.f4096f;
                        c.c.a.n.b bVar2 = new c.c.a.n.b();
                        h hVar = new h();
                        Context applicationContext = context.getApplicationContext();
                        Objects.requireNonNull((a) bVar);
                        this.f4092b = new c.c.a.i(b2, bVar2, hVar, applicationContext);
                    }
                }
            }
            return this.f4092b;
        }
        throw new IllegalArgumentException("You cannot start a load on a null Context");
    }

    public c.c.a.i d(b.q.b.d dVar) {
        if (c.c.a.s.j.h()) {
            return c(dVar.getApplicationContext());
        }
        if (!dVar.isDestroyed()) {
            this.f4097g.a(dVar);
            b.q.b.q m = dVar.m();
            boolean g2 = g(dVar);
            s f2 = f(m, null);
            c.c.a.i iVar = f2.f4105f;
            if (iVar == null) {
                c.c.a.b b2 = c.c.a.b.b(dVar);
                b bVar = this.f4096f;
                c.c.a.n.a aVar = f2.f4101b;
                q qVar = f2.f4102c;
                Objects.requireNonNull((a) bVar);
                c.c.a.i iVar2 = new c.c.a.i(b2, aVar, qVar, dVar);
                if (g2) {
                    iVar2.onStart();
                }
                f2.f4105f = iVar2;
                return iVar2;
            }
            return iVar;
        }
        throw new IllegalArgumentException("You cannot start a load for a destroyed activity");
    }

    public final o e(FragmentManager fragmentManager, Fragment fragment) {
        o oVar = (o) fragmentManager.findFragmentByTag("com.bumptech.glide.manager");
        if (oVar == null && (oVar = this.f4093c.get(fragmentManager)) == null) {
            oVar = new o();
            oVar.f4089g = fragment;
            if (fragment != null && fragment.getActivity() != null) {
                oVar.a(fragment.getActivity());
            }
            this.f4093c.put(fragmentManager, oVar);
            fragmentManager.beginTransaction().add(oVar, "com.bumptech.glide.manager").commitAllowingStateLoss();
            this.f4095e.obtainMessage(1, fragmentManager).sendToTarget();
        }
        return oVar;
    }

    public final s f(b.q.b.q qVar, androidx.fragment.app.Fragment fragment) {
        s sVar = (s) qVar.I("com.bumptech.glide.manager");
        if (sVar == null && (sVar = this.f4094d.get(qVar)) == null) {
            sVar = new s();
            sVar.f4106g = fragment;
            if (fragment != null && fragment.getContext() != null) {
                androidx.fragment.app.Fragment fragment2 = fragment;
                while (fragment2.getParentFragment() != null) {
                    fragment2 = fragment2.getParentFragment();
                }
                b.q.b.q fragmentManager = fragment2.getFragmentManager();
                if (fragmentManager != null) {
                    sVar.c(fragment.getContext(), fragmentManager);
                }
            }
            this.f4094d.put(qVar, sVar);
            b.q.b.a aVar = new b.q.b.a(qVar);
            aVar.d(0, sVar, "com.bumptech.glide.manager", 1);
            aVar.f();
            this.f4095e.obtainMessage(2, qVar).sendToTarget();
        }
        return sVar;
    }

    @Override // android.os.Handler.Callback
    public boolean handleMessage(Message message) {
        Object obj;
        Object remove;
        Object obj2;
        int i = message.what;
        Object obj3 = null;
        boolean z = true;
        if (i == 1) {
            obj = (FragmentManager) message.obj;
            remove = this.f4093c.remove(obj);
        } else if (i == 2) {
            obj = (b.q.b.q) message.obj;
            remove = this.f4094d.remove(obj);
        } else {
            z = false;
            obj2 = null;
            if (z && obj3 == null && Log.isLoggable("RMRetriever", 5)) {
                Log.w("RMRetriever", "Failed to remove expected request manager fragment, manager: " + obj2);
            }
            return z;
        }
        Object obj4 = obj;
        obj3 = remove;
        obj2 = obj4;
        if (z) {
            Log.w("RMRetriever", "Failed to remove expected request manager fragment, manager: " + obj2);
        }
        return z;
    }
}