package b.q.b;

import android.os.Bundle;
import android.os.Parcelable;
import android.util.Log;
import android.util.SparseArray;
import androidx.fragment.app.Fragment;
import b.t.e;

/* compiled from: FragmentStateManager.java */
/* loaded from: classes.dex */
public class w {

    /* renamed from: a  reason: collision with root package name */
    public final p f2536a;

    /* renamed from: b  reason: collision with root package name */
    public final Fragment f2537b;

    /* renamed from: c  reason: collision with root package name */
    public int f2538c = -1;

    public w(p pVar, Fragment fragment) {
        this.f2536a = pVar;
        this.f2537b = fragment;
    }

    public void a(ClassLoader classLoader) {
        Bundle bundle = this.f2537b.mSavedFragmentState;
        if (bundle == null) {
            return;
        }
        bundle.setClassLoader(classLoader);
        Fragment fragment = this.f2537b;
        fragment.mSavedViewState = fragment.mSavedFragmentState.getSparseParcelableArray("android:view_state");
        Fragment fragment2 = this.f2537b;
        fragment2.mTargetWho = fragment2.mSavedFragmentState.getString("android:target_state");
        Fragment fragment3 = this.f2537b;
        if (fragment3.mTargetWho != null) {
            fragment3.mTargetRequestCode = fragment3.mSavedFragmentState.getInt("android:target_req_state", 0);
        }
        Fragment fragment4 = this.f2537b;
        Boolean bool = fragment4.mSavedUserVisibleHint;
        if (bool != null) {
            fragment4.mUserVisibleHint = bool.booleanValue();
            this.f2537b.mSavedUserVisibleHint = null;
        } else {
            fragment4.mUserVisibleHint = fragment4.mSavedFragmentState.getBoolean("android:user_visible_hint", true);
        }
        Fragment fragment5 = this.f2537b;
        if (fragment5.mUserVisibleHint) {
            return;
        }
        fragment5.mDeferStart = true;
    }

    public void b() {
        if (this.f2537b.mView == null) {
            return;
        }
        SparseArray<Parcelable> sparseArray = new SparseArray<>();
        this.f2537b.mView.saveHierarchyState(sparseArray);
        if (sparseArray.size() > 0) {
            this.f2537b.mSavedViewState = sparseArray;
        }
    }

    public w(p pVar, ClassLoader classLoader, m mVar, v vVar) {
        this.f2536a = pVar;
        Fragment a2 = mVar.a(classLoader, vVar.f2529b);
        this.f2537b = a2;
        Bundle bundle = vVar.k;
        if (bundle != null) {
            bundle.setClassLoader(classLoader);
        }
        a2.setArguments(vVar.k);
        a2.mWho = vVar.f2530c;
        a2.mFromLayout = vVar.f2531d;
        a2.mRestored = true;
        a2.mFragmentId = vVar.f2532e;
        a2.mContainerId = vVar.f2533f;
        a2.mTag = vVar.f2534g;
        a2.mRetainInstance = vVar.f2535h;
        a2.mRemoving = vVar.i;
        a2.mDetached = vVar.j;
        a2.mHidden = vVar.l;
        a2.mMaxState = e.b.values()[vVar.m];
        Bundle bundle2 = vVar.n;
        if (bundle2 != null) {
            a2.mSavedFragmentState = bundle2;
        } else {
            a2.mSavedFragmentState = new Bundle();
        }
        if (q.N(2)) {
            Log.v("FragmentManager", "Instantiated fragment " + a2);
        }
    }

    public w(p pVar, Fragment fragment, v vVar) {
        this.f2536a = pVar;
        this.f2537b = fragment;
        fragment.mSavedViewState = null;
        fragment.mBackStackNesting = 0;
        fragment.mInLayout = false;
        fragment.mAdded = false;
        Fragment fragment2 = fragment.mTarget;
        fragment.mTargetWho = fragment2 != null ? fragment2.mWho : null;
        fragment.mTarget = null;
        Bundle bundle = vVar.n;
        if (bundle != null) {
            fragment.mSavedFragmentState = bundle;
        } else {
            fragment.mSavedFragmentState = new Bundle();
        }
    }
}