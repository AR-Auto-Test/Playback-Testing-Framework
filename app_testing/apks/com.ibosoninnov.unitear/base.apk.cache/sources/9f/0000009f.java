package androidx.databinding.library.baseAdapters;

import android.util.SparseIntArray;
import android.view.View;
import androidx.databinding.ViewDataBinding;
import b.m.d;
import b.m.e;
import java.util.ArrayList;
import java.util.List;

/* loaded from: classes.dex */
public class DataBinderMapperImpl extends d {

    /* renamed from: a  reason: collision with root package name */
    public static final SparseIntArray f270a = new SparseIntArray(0);

    @Override // b.m.d
    public List<d> a() {
        return new ArrayList(0);
    }

    @Override // b.m.d
    public ViewDataBinding b(e eVar, View view, int i) {
        if (f270a.get(i) <= 0 || view.getTag() != null) {
            return null;
        }
        throw new RuntimeException("view must have a tag");
    }

    @Override // b.m.d
    public ViewDataBinding c(e eVar, View[] viewArr, int i) {
        if (viewArr == null || viewArr.length == 0 || f270a.get(i) <= 0 || viewArr[0].getTag() != null) {
            return null;
        }
        throw new RuntimeException("view must have a tag");
    }
}