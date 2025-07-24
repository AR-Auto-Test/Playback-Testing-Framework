package c.c.a.n;

import android.content.Context;
import android.util.Log;
import androidx.fragment.app.Fragment;
import java.util.HashSet;
import java.util.Set;

/* compiled from: SupportRequestManagerFragment.java */
/* loaded from: classes.dex */
public class s extends Fragment {

    /* renamed from: b  reason: collision with root package name */
    public final c.c.a.n.a f4101b;

    /* renamed from: c  reason: collision with root package name */
    public final q f4102c;

    /* renamed from: d  reason: collision with root package name */
    public final Set<s> f4103d;

    /* renamed from: e  reason: collision with root package name */
    public s f4104e;

    /* renamed from: f  reason: collision with root package name */
    public c.c.a.i f4105f;

    /* renamed from: g  reason: collision with root package name */
    public Fragment f4106g;

    /* compiled from: SupportRequestManagerFragment.java */
    /* loaded from: classes.dex */
    public class a implements q {
        public a() {
        }

        public String toString() {
            return super.toString() + "{fragment=" + s.this + "}";
        }
    }

    public s() {
        c.c.a.n.a aVar = new c.c.a.n.a();
        this.f4102c = new a();
        this.f4103d = new HashSet();
        this.f4101b = aVar;
    }

    public final Fragment a() {
        Fragment parentFragment = getParentFragment();
        return parentFragment != null ? parentFragment : this.f4106g;
    }

    public final void c(Context context, b.q.b.q qVar) {
        d();
        s f2 = c.c.a.b.b(context).i.f(qVar, null);
        this.f4104e = f2;
        if (equals(f2)) {
            return;
        }
        this.f4104e.f4103d.add(this);
    }

    public final void d() {
        s sVar = this.f4104e;
        if (sVar != null) {
            sVar.f4103d.remove(this);
            this.f4104e = null;
        }
    }

    @Override // androidx.fragment.app.Fragment
    public void onAttach(Context context) {
        super.onAttach(context);
        Fragment fragment = this;
        while (fragment.getParentFragment() != null) {
            fragment = fragment.getParentFragment();
        }
        b.q.b.q fragmentManager = fragment.getFragmentManager();
        if (fragmentManager == null) {
            if (Log.isLoggable("SupportRMFragment", 5)) {
                Log.w("SupportRMFragment", "Unable to register fragment with root, ancestor detached");
                return;
            }
            return;
        }
        try {
            c(getContext(), fragmentManager);
        } catch (IllegalStateException e2) {
            if (Log.isLoggable("SupportRMFragment", 5)) {
                Log.w("SupportRMFragment", "Unable to register fragment with root", e2);
            }
        }
    }

    @Override // androidx.fragment.app.Fragment
    public void onDestroy() {
        super.onDestroy();
        this.f4101b.c();
        d();
    }

    @Override // androidx.fragment.app.Fragment
    public void onDetach() {
        super.onDetach();
        this.f4106g = null;
        d();
    }

    @Override // androidx.fragment.app.Fragment
    public void onStart() {
        super.onStart();
        this.f4101b.d();
    }

    @Override // androidx.fragment.app.Fragment
    public void onStop() {
        super.onStop();
        this.f4101b.e();
    }

    @Override // androidx.fragment.app.Fragment
    public String toString() {
        return super.toString() + "{parent=" + a() + "}";
    }
}