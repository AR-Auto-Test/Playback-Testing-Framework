package b.m;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import androidx.databinding.DataBinderMapperImpl;
import androidx.databinding.ViewDataBinding;

/* compiled from: DataBindingUtil.java */
/* loaded from: classes.dex */
public class f {

    /* renamed from: a  reason: collision with root package name */
    public static d f2334a = new DataBinderMapperImpl();

    public static <T extends ViewDataBinding> T a(e eVar, View view, int i) {
        return (T) f2334a.b(eVar, view, i);
    }

    public static <T extends ViewDataBinding> T b(LayoutInflater layoutInflater, int i, ViewGroup viewGroup, boolean z) {
        boolean z2 = z;
        int childCount = z2 ? viewGroup.getChildCount() : 0;
        View inflate = layoutInflater.inflate(i, viewGroup, z);
        if (z2) {
            int childCount2 = viewGroup.getChildCount();
            int i2 = childCount2 - childCount;
            if (i2 == 1) {
                return (T) a(null, viewGroup.getChildAt(childCount2 - 1), i);
            }
            View[] viewArr = new View[i2];
            for (int i3 = 0; i3 < i2; i3++) {
                viewArr[i3] = viewGroup.getChildAt(i3 + childCount);
            }
            return (T) f2334a.c(null, viewArr, i);
        }
        return (T) a(null, inflate, i);
    }
}