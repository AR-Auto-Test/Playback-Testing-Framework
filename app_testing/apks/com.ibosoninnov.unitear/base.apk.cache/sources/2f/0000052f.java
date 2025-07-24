package b.q.b;

import android.content.Context;
import android.content.res.TypedArray;
import android.util.AttributeSet;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import androidx.fragment.app.Fragment;

/* compiled from: FragmentLayoutInflaterFactory.java */
/* loaded from: classes.dex */
public class o implements LayoutInflater.Factory2 {

    /* renamed from: b  reason: collision with root package name */
    public final q f2493b;

    public o(q qVar) {
        this.f2493b = qVar;
    }

    @Override // android.view.LayoutInflater.Factory
    public View onCreateView(String str, Context context, AttributeSet attributeSet) {
        return onCreateView(null, str, context, attributeSet);
    }

    @Override // android.view.LayoutInflater.Factory2
    public View onCreateView(View view, String str, Context context, AttributeSet attributeSet) {
        boolean z;
        if (k.class.getName().equals(str)) {
            return new k(context, attributeSet, this.f2493b);
        }
        if ("fragment".equals(str)) {
            String attributeValue = attributeSet.getAttributeValue(null, "class");
            TypedArray obtainStyledAttributes = context.obtainStyledAttributes(attributeSet, b.q.a.f2395a);
            if (attributeValue == null) {
                attributeValue = obtainStyledAttributes.getString(0);
            }
            int resourceId = obtainStyledAttributes.getResourceId(1, -1);
            String string = obtainStyledAttributes.getString(2);
            obtainStyledAttributes.recycle();
            if (attributeValue != null) {
                ClassLoader classLoader = context.getClassLoader();
                b.f.h<String, Class<?>> hVar = m.f2488a;
                try {
                    z = Fragment.class.isAssignableFrom(m.b(classLoader, attributeValue));
                } catch (ClassNotFoundException unused) {
                    z = false;
                }
                if (z) {
                    int id = view != null ? view.getId() : 0;
                    if (id == -1 && resourceId == -1 && string == null) {
                        throw new IllegalArgumentException(attributeSet.getPositionDescription() + ": Must specify unique android:id, android:tag, or have a parent with an id for " + attributeValue);
                    }
                    Fragment H = resourceId != -1 ? this.f2493b.H(resourceId) : null;
                    if (H == null && string != null) {
                        H = this.f2493b.I(string);
                    }
                    if (H == null && id != -1) {
                        H = this.f2493b.H(id);
                    }
                    if (q.N(2)) {
                        StringBuilder x = c.b.a.a.a.x("onCreateView: id=0x");
                        x.append(Integer.toHexString(resourceId));
                        x.append(" fname=");
                        x.append(attributeValue);
                        x.append(" existing=");
                        x.append(H);
                        Log.v("FragmentManager", x.toString());
                    }
                    if (H == null) {
                        H = this.f2493b.L().a(context.getClassLoader(), attributeValue);
                        H.mFromLayout = true;
                        H.mFragmentId = resourceId != 0 ? resourceId : id;
                        H.mContainerId = id;
                        H.mTag = string;
                        H.mInLayout = true;
                        q qVar = this.f2493b;
                        H.mFragmentManager = qVar;
                        n<?> nVar = qVar.n;
                        H.mHost = nVar;
                        H.onInflate(nVar.f2490c, attributeSet, H.mSavedFragmentState);
                        this.f2493b.b(H);
                        q qVar2 = this.f2493b;
                        qVar2.U(H, qVar2.m);
                    } else if (!H.mInLayout) {
                        H.mInLayout = true;
                        n<?> nVar2 = this.f2493b.n;
                        H.mHost = nVar2;
                        H.onInflate(nVar2.f2490c, attributeSet, H.mSavedFragmentState);
                    } else {
                        throw new IllegalArgumentException(attributeSet.getPositionDescription() + ": Duplicate id 0x" + Integer.toHexString(resourceId) + ", tag " + string + ", or parent id 0x" + Integer.toHexString(id) + " with another fragment for " + attributeValue);
                    }
                    q qVar3 = this.f2493b;
                    int i = qVar3.m;
                    if (i < 1 && H.mFromLayout) {
                        qVar3.U(H, 1);
                    } else {
                        qVar3.U(H, i);
                    }
                    View view2 = H.mView;
                    if (view2 != null) {
                        if (resourceId != 0) {
                            view2.setId(resourceId);
                        }
                        if (H.mView.getTag() == null) {
                            H.mView.setTag(string);
                        }
                        return H.mView;
                    }
                    throw new IllegalStateException(c.b.a.a.a.r("Fragment ", attributeValue, " did not create a view."));
                }
            }
            return null;
        }
        return null;
    }
}