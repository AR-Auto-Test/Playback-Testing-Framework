package androidx.navigation.fragment;

import android.app.Dialog;
import android.content.Context;
import android.content.res.TypedArray;
import android.os.Bundle;
import android.util.AttributeSet;
import android.util.Log;
import android.view.View;
import androidx.fragment.app.Fragment;
import androidx.navigation.NavController;
import b.j.b.d;
import b.q.b.c;
import b.t.e;
import b.t.f;
import b.t.h;
import b.t.i;
import b.v.j;
import b.v.o;
import b.v.q;
import b.v.u.b;
import java.util.HashSet;

@q.b("dialog")
/* loaded from: classes.dex */
public final class DialogFragmentNavigator extends q<a> {

    /* renamed from: a  reason: collision with root package name */
    public final Context f352a;

    /* renamed from: b  reason: collision with root package name */
    public final b.q.b.q f353b;

    /* renamed from: c  reason: collision with root package name */
    public int f354c = 0;

    /* renamed from: d  reason: collision with root package name */
    public final HashSet<String> f355d = new HashSet<>();

    /* renamed from: e  reason: collision with root package name */
    public f f356e = new f(this) { // from class: androidx.navigation.fragment.DialogFragmentNavigator.1
        @Override // b.t.f
        public void e(h hVar, e.a aVar) {
            NavController t;
            if (aVar == e.a.ON_STOP) {
                c cVar = (c) hVar;
                if (cVar.requireDialog().isShowing()) {
                    return;
                }
                int i = b.f2691b;
                Fragment fragment = cVar;
                while (true) {
                    if (fragment != null) {
                        if (fragment instanceof b) {
                            t = ((b) fragment).f2692c;
                            if (t == null) {
                                throw new IllegalStateException("NavController is not available before onCreate()");
                            }
                        } else {
                            Fragment fragment2 = fragment.getParentFragmentManager().q;
                            if (fragment2 instanceof b) {
                                t = ((b) fragment2).f2692c;
                                if (t == null) {
                                    throw new IllegalStateException("NavController is not available before onCreate()");
                                }
                            } else {
                                fragment = fragment.getParentFragment();
                            }
                        }
                    } else {
                        View view = cVar.getView();
                        if (view != null) {
                            t = d.t(view);
                        } else {
                            Dialog dialog = cVar.getDialog();
                            if (dialog != null && dialog.getWindow() != null) {
                                t = d.t(dialog.getWindow().getDecorView());
                            } else {
                                throw new IllegalStateException("Fragment " + cVar + " does not have a NavController set");
                            }
                        }
                    }
                }
                t.e();
            }
        }
    };

    /* loaded from: classes.dex */
    public static class a extends j implements b.v.b {
        public String j;

        public a(q<? extends a> qVar) {
            super(qVar);
        }

        @Override // b.v.j
        public void d(Context context, AttributeSet attributeSet) {
            super.d(context, attributeSet);
            TypedArray obtainAttributes = context.getResources().obtainAttributes(attributeSet, b.v.u.d.f2698a);
            String string = obtainAttributes.getString(0);
            if (string != null) {
                this.j = string;
            }
            obtainAttributes.recycle();
        }
    }

    public DialogFragmentNavigator(Context context, b.q.b.q qVar) {
        this.f352a = context;
        this.f353b = qVar;
    }

    /* JADX DEBUG: Return type fixed from 'b.v.j' to match base method */
    @Override // b.v.q
    public a a() {
        return new a(this);
    }

    /* JADX DEBUG: Method arguments types fixed to match base method, original types: [b.v.j, android.os.Bundle, b.v.o, b.v.q$a] */
    @Override // b.v.q
    public j b(a aVar, Bundle bundle, o oVar, q.a aVar2) {
        a aVar3 = aVar;
        if (this.f353b.Q()) {
            Log.i("DialogFragmentNavigator", "Ignoring navigate() call: FragmentManager has already saved its state");
            return null;
        }
        String str = aVar3.j;
        if (str != null) {
            if (str.charAt(0) == '.') {
                str = this.f352a.getPackageName() + str;
            }
            Fragment a2 = this.f353b.L().a(this.f352a.getClassLoader(), str);
            if (!c.class.isAssignableFrom(a2.getClass())) {
                StringBuilder x = c.b.a.a.a.x("Dialog destination ");
                String str2 = aVar3.j;
                if (str2 != null) {
                    throw new IllegalArgumentException(c.b.a.a.a.v(x, str2, " is not an instance of DialogFragment"));
                }
                throw new IllegalStateException("DialogFragment class was not set");
            }
            c cVar = (c) a2;
            cVar.setArguments(bundle);
            cVar.getLifecycle().a(this.f356e);
            b.q.b.q qVar = this.f353b;
            StringBuilder x2 = c.b.a.a.a.x("androidx-nav-fragment:navigator:dialog:");
            int i = this.f354c;
            this.f354c = i + 1;
            x2.append(i);
            cVar.show(qVar, x2.toString());
            return aVar3;
        }
        throw new IllegalStateException("DialogFragment class was not set");
    }

    @Override // b.v.q
    public void c(Bundle bundle) {
        this.f354c = bundle.getInt("androidx-nav-dialogfragment:navigator:count", 0);
        for (int i = 0; i < this.f354c; i++) {
            b.q.b.q qVar = this.f353b;
            c cVar = (c) qVar.I("androidx-nav-fragment:navigator:dialog:" + i);
            if (cVar != null) {
                cVar.getLifecycle().a(this.f356e);
            } else {
                HashSet<String> hashSet = this.f355d;
                hashSet.add("androidx-nav-fragment:navigator:dialog:" + i);
            }
        }
    }

    @Override // b.v.q
    public Bundle d() {
        if (this.f354c == 0) {
            return null;
        }
        Bundle bundle = new Bundle();
        bundle.putInt("androidx-nav-dialogfragment:navigator:count", this.f354c);
        return bundle;
    }

    @Override // b.v.q
    public boolean e() {
        if (this.f354c == 0) {
            return false;
        }
        if (this.f353b.Q()) {
            Log.i("DialogFragmentNavigator", "Ignoring popBackStack() call: FragmentManager has already saved its state");
            return false;
        }
        b.q.b.q qVar = this.f353b;
        StringBuilder x = c.b.a.a.a.x("androidx-nav-fragment:navigator:dialog:");
        int i = this.f354c - 1;
        this.f354c = i;
        x.append(i);
        Fragment I = qVar.I(x.toString());
        if (I != null) {
            e lifecycle = I.getLifecycle();
            ((i) lifecycle).f2578a.e(this.f356e);
            ((c) I).dismiss();
        }
        return true;
    }
}