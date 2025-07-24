package b.v.u;

import android.content.Context;
import android.content.res.TypedArray;
import android.os.Bundle;
import android.os.Parcelable;
import android.util.AttributeSet;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import androidx.activity.OnBackPressedDispatcher;
import androidx.fragment.app.Fragment;
import androidx.navigation.fragment.DialogFragmentNavigator;
import b.q.b.k;
import b.q.b.q;
import b.t.i;
import b.t.y;
import b.v.e;
import b.v.f;
import b.v.g;
import b.v.j;
import b.v.m;
import b.v.r;
import b.v.s;
import com.ibosoninnov.unitear.R;
import java.util.ArrayList;
import java.util.Map;
import java.util.Objects;

/* compiled from: NavHostFragment.java */
/* loaded from: classes.dex */
public class b extends Fragment {

    /* renamed from: b  reason: collision with root package name */
    public static final /* synthetic */ int f2691b = 0;

    /* renamed from: c  reason: collision with root package name */
    public m f2692c;

    /* renamed from: d  reason: collision with root package name */
    public Boolean f2693d = null;

    /* renamed from: e  reason: collision with root package name */
    public View f2694e;

    /* renamed from: f  reason: collision with root package name */
    public int f2695f;

    /* renamed from: g  reason: collision with root package name */
    public boolean f2696g;

    @Override // androidx.fragment.app.Fragment
    public void onAttach(Context context) {
        super.onAttach(context);
        if (this.f2696g) {
            b.q.b.a aVar = new b.q.b.a(getParentFragmentManager());
            aVar.o(this);
            aVar.c();
        }
    }

    @Override // androidx.fragment.app.Fragment
    public void onAttachFragment(Fragment fragment) {
        super.onAttachFragment(fragment);
        r rVar = this.f2692c.k;
        Objects.requireNonNull(rVar);
        DialogFragmentNavigator dialogFragmentNavigator = (DialogFragmentNavigator) rVar.c(r.b(DialogFragmentNavigator.class));
        if (dialogFragmentNavigator.f355d.remove(fragment.getTag())) {
            fragment.getLifecycle().a(dialogFragmentNavigator.f356e);
        }
    }

    @Override // androidx.fragment.app.Fragment
    public void onCreate(Bundle bundle) {
        Bundle bundle2;
        m mVar = new m(requireContext());
        this.f2692c = mVar;
        if (this != mVar.i) {
            mVar.i = this;
            getLifecycle().a(mVar.m);
        }
        m mVar2 = this.f2692c;
        OnBackPressedDispatcher onBackPressedDispatcher = requireActivity().f41f;
        if (mVar2.i != null) {
            mVar2.n.b();
            onBackPressedDispatcher.a(mVar2.i, mVar2.n);
            ((i) mVar2.i.getLifecycle()).f2578a.e(mVar2.m);
            mVar2.i.getLifecycle().a(mVar2.m);
            m mVar3 = this.f2692c;
            Boolean bool = this.f2693d;
            mVar3.o = bool != null && bool.booleanValue();
            mVar3.h();
            this.f2693d = null;
            m mVar4 = this.f2692c;
            y viewModelStore = getViewModelStore();
            if (mVar4.j != g.c(viewModelStore)) {
                if (mVar4.f349h.isEmpty()) {
                    mVar4.j = g.c(viewModelStore);
                } else {
                    throw new IllegalStateException("ViewModelStore should be set before setGraph call");
                }
            }
            m mVar5 = this.f2692c;
            mVar5.k.a(new DialogFragmentNavigator(requireContext(), getChildFragmentManager()));
            r rVar = mVar5.k;
            Context requireContext = requireContext();
            q childFragmentManager = getChildFragmentManager();
            int id = getId();
            if (id == 0 || id == -1) {
                id = R.id.nav_host_fragment_container;
            }
            rVar.a(new a(requireContext, childFragmentManager, id));
            if (bundle != null) {
                bundle2 = bundle.getBundle("android-support-nav:fragment:navControllerState");
                if (bundle.getBoolean("android-support-nav:fragment:defaultHost", false)) {
                    this.f2696g = true;
                    b.q.b.a aVar = new b.q.b.a(getParentFragmentManager());
                    aVar.o(this);
                    aVar.c();
                }
                this.f2695f = bundle.getInt("android-support-nav:fragment:graphId");
            } else {
                bundle2 = null;
            }
            if (bundle2 != null) {
                m mVar6 = this.f2692c;
                Objects.requireNonNull(mVar6);
                bundle2.setClassLoader(mVar6.f342a.getClassLoader());
                mVar6.f346e = bundle2.getBundle("android-support-nav:controller:navigatorState");
                mVar6.f347f = bundle2.getParcelableArray("android-support-nav:controller:backStack");
                mVar6.f348g = bundle2.getBoolean("android-support-nav:controller:deepLinkHandled");
            }
            int i = this.f2695f;
            if (i != 0) {
                this.f2692c.g(i, null);
            } else {
                Bundle arguments = getArguments();
                int i2 = arguments != null ? arguments.getInt("android-support-nav:fragment:graphId") : 0;
                Bundle bundle3 = arguments != null ? arguments.getBundle("android-support-nav:fragment:startDestinationArgs") : null;
                if (i2 != 0) {
                    this.f2692c.g(i2, bundle3);
                }
            }
            super.onCreate(bundle);
            return;
        }
        throw new IllegalStateException("You must call setLifecycleOwner() before calling setOnBackPressedDispatcher()");
    }

    @Override // androidx.fragment.app.Fragment
    public View onCreateView(LayoutInflater layoutInflater, ViewGroup viewGroup, Bundle bundle) {
        k kVar = new k(layoutInflater.getContext());
        int id = getId();
        if (id == 0 || id == -1) {
            id = R.id.nav_host_fragment_container;
        }
        kVar.setId(id);
        return kVar;
    }

    @Override // androidx.fragment.app.Fragment
    public void onDestroyView() {
        super.onDestroyView();
        View view = this.f2694e;
        if (view != null && b.j.b.d.t(view) == this.f2692c) {
            this.f2694e.setTag(R.id.nav_controller_view_tag, null);
        }
        this.f2694e = null;
    }

    @Override // androidx.fragment.app.Fragment
    public void onInflate(Context context, AttributeSet attributeSet, Bundle bundle) {
        super.onInflate(context, attributeSet, bundle);
        TypedArray obtainStyledAttributes = context.obtainStyledAttributes(attributeSet, s.f2680b);
        int resourceId = obtainStyledAttributes.getResourceId(0, 0);
        if (resourceId != 0) {
            this.f2695f = resourceId;
        }
        obtainStyledAttributes.recycle();
        TypedArray obtainStyledAttributes2 = context.obtainStyledAttributes(attributeSet, d.f2700c);
        if (obtainStyledAttributes2.getBoolean(0, false)) {
            this.f2696g = true;
        }
        obtainStyledAttributes2.recycle();
    }

    @Override // androidx.fragment.app.Fragment
    public void onPrimaryNavigationFragmentChanged(boolean z) {
        m mVar = this.f2692c;
        if (mVar != null) {
            mVar.o = z;
            mVar.h();
            return;
        }
        this.f2693d = Boolean.valueOf(z);
    }

    @Override // androidx.fragment.app.Fragment
    public void onSaveInstanceState(Bundle bundle) {
        Bundle bundle2;
        super.onSaveInstanceState(bundle);
        m mVar = this.f2692c;
        Objects.requireNonNull(mVar);
        ArrayList<String> arrayList = new ArrayList<>();
        Bundle bundle3 = new Bundle();
        for (Map.Entry<String, b.v.q<? extends j>> entry : mVar.k.f2678b.entrySet()) {
            String key = entry.getKey();
            Bundle d2 = entry.getValue().d();
            if (d2 != null) {
                arrayList.add(key);
                bundle3.putBundle(key, d2);
            }
        }
        if (arrayList.isEmpty()) {
            bundle2 = null;
        } else {
            bundle2 = new Bundle();
            bundle3.putStringArrayList("android-support-nav:controller:navigatorState:names", arrayList);
            bundle2.putBundle("android-support-nav:controller:navigatorState", bundle3);
        }
        if (!mVar.f349h.isEmpty()) {
            if (bundle2 == null) {
                bundle2 = new Bundle();
            }
            Parcelable[] parcelableArr = new Parcelable[mVar.f349h.size()];
            int i = 0;
            for (e eVar : mVar.f349h) {
                parcelableArr[i] = new f(eVar);
                i++;
            }
            bundle2.putParcelableArray("android-support-nav:controller:backStack", parcelableArr);
        }
        if (mVar.f348g) {
            if (bundle2 == null) {
                bundle2 = new Bundle();
            }
            bundle2.putBoolean("android-support-nav:controller:deepLinkHandled", mVar.f348g);
        }
        if (bundle2 != null) {
            bundle.putBundle("android-support-nav:fragment:navControllerState", bundle2);
        }
        if (this.f2696g) {
            bundle.putBoolean("android-support-nav:fragment:defaultHost", true);
        }
        int i2 = this.f2695f;
        if (i2 != 0) {
            bundle.putInt("android-support-nav:fragment:graphId", i2);
        }
    }

    @Override // androidx.fragment.app.Fragment
    public void onViewCreated(View view, Bundle bundle) {
        super.onViewCreated(view, bundle);
        if (view instanceof ViewGroup) {
            view.setTag(R.id.nav_controller_view_tag, this.f2692c);
            if (view.getParent() != null) {
                View view2 = (View) view.getParent();
                this.f2694e = view2;
                if (view2.getId() == getId()) {
                    this.f2694e.setTag(R.id.nav_controller_view_tag, this.f2692c);
                    return;
                }
                return;
            }
            return;
        }
        throw new IllegalStateException("created host view " + view + " is not a ViewGroup");
    }
}