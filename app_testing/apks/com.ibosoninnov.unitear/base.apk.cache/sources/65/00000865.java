package c.c.a.n;

import android.app.Activity;
import android.app.Fragment;
import android.util.Log;
import java.util.HashSet;
import java.util.Objects;
import java.util.Set;

/* compiled from: RequestManagerFragment.java */
@Deprecated
/* loaded from: classes.dex */
public class o extends Fragment {

    /* renamed from: b  reason: collision with root package name */
    public final c.c.a.n.a f4084b;

    /* renamed from: c  reason: collision with root package name */
    public final q f4085c;

    /* renamed from: d  reason: collision with root package name */
    public final Set<o> f4086d;

    /* renamed from: e  reason: collision with root package name */
    public c.c.a.i f4087e;

    /* renamed from: f  reason: collision with root package name */
    public o f4088f;

    /* renamed from: g  reason: collision with root package name */
    public Fragment f4089g;

    /* compiled from: RequestManagerFragment.java */
    /* loaded from: classes.dex */
    public class a implements q {
        public a() {
        }

        public String toString() {
            return super.toString() + "{fragment=" + o.this + "}";
        }
    }

    public o() {
        c.c.a.n.a aVar = new c.c.a.n.a();
        this.f4085c = new a();
        this.f4086d = new HashSet();
        this.f4084b = aVar;
    }

    public final void a(Activity activity) {
        b();
        p pVar = c.c.a.b.b(activity).i;
        Objects.requireNonNull(pVar);
        o e2 = pVar.e(activity.getFragmentManager(), null);
        this.f4088f = e2;
        if (equals(e2)) {
            return;
        }
        this.f4088f.f4086d.add(this);
    }

    public final void b() {
        o oVar = this.f4088f;
        if (oVar != null) {
            oVar.f4086d.remove(this);
            this.f4088f = null;
        }
    }

    @Override // android.app.Fragment
    public void onAttach(Activity activity) {
        super.onAttach(activity);
        try {
            a(activity);
        } catch (IllegalStateException e2) {
            if (Log.isLoggable("RMFragment", 5)) {
                Log.w("RMFragment", "Unable to register fragment with root", e2);
            }
        }
    }

    @Override // android.app.Fragment
    public void onDestroy() {
        super.onDestroy();
        this.f4084b.c();
        b();
    }

    @Override // android.app.Fragment
    public void onDetach() {
        super.onDetach();
        b();
    }

    @Override // android.app.Fragment
    public void onStart() {
        super.onStart();
        this.f4084b.d();
    }

    @Override // android.app.Fragment
    public void onStop() {
        super.onStop();
        this.f4084b.e();
    }

    @Override // android.app.Fragment
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(super.toString());
        sb.append("{parent=");
        Fragment parentFragment = getParentFragment();
        if (parentFragment == null) {
            parentFragment = this.f4089g;
        }
        sb.append(parentFragment);
        sb.append("}");
        return sb.toString();
    }
}