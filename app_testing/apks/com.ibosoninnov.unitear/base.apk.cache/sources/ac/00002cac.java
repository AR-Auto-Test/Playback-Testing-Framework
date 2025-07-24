package com.ibosoninnov.unitear;

import android.util.SparseIntArray;
import android.view.View;
import androidx.databinding.ViewDataBinding;
import b.m.d;
import b.m.e;
import c.b.a.a.a;
import c.e.b.ff.b;
import c.e.b.ff.f;
import c.e.b.ff.h;
import c.e.b.ff.j;
import java.util.ArrayList;
import java.util.List;

/* loaded from: classes2.dex */
public class DataBinderMapperImpl extends d {

    /* renamed from: a  reason: collision with root package name */
    public static final SparseIntArray f5672a;

    static {
        SparseIntArray sparseIntArray = new SparseIntArray(5);
        f5672a = sparseIntArray;
        sparseIntArray.put(R.layout.item_about, 1);
        sparseIntArray.put(R.layout.item_ar_gallery, 2);
        sparseIntArray.put(R.layout.item_arobjects, 3);
        sparseIntArray.put(R.layout.item_history, 4);
        sparseIntArray.put(R.layout.item_menu, 5);
    }

    @Override // b.m.d
    public List<d> a() {
        ArrayList arrayList = new ArrayList(1);
        arrayList.add(new androidx.databinding.library.baseAdapters.DataBinderMapperImpl());
        return arrayList;
    }

    @Override // b.m.d
    public ViewDataBinding b(e eVar, View view, int i) {
        int i2 = f5672a.get(i);
        if (i2 > 0) {
            Object tag = view.getTag();
            if (tag != null) {
                if (i2 == 1) {
                    if ("layout/item_about_0".equals(tag)) {
                        return new b(eVar, view);
                    }
                    throw new IllegalArgumentException(a.p("The tag for item_about is invalid. Received: ", tag));
                } else if (i2 == 2) {
                    if ("layout/item_ar_gallery_0".equals(tag)) {
                        return new c.e.b.ff.d(eVar, view);
                    }
                    throw new IllegalArgumentException(a.p("The tag for item_ar_gallery is invalid. Received: ", tag));
                } else if (i2 == 3) {
                    if ("layout/item_arobjects_0".equals(tag)) {
                        return new f(eVar, view);
                    }
                    throw new IllegalArgumentException(a.p("The tag for item_arobjects is invalid. Received: ", tag));
                } else if (i2 == 4) {
                    if ("layout/item_history_0".equals(tag)) {
                        return new h(eVar, view);
                    }
                    throw new IllegalArgumentException(a.p("The tag for item_history is invalid. Received: ", tag));
                } else if (i2 != 5) {
                    return null;
                } else {
                    if ("layout/item_menu_0".equals(tag)) {
                        return new j(eVar, view);
                    }
                    throw new IllegalArgumentException(a.p("The tag for item_menu is invalid. Received: ", tag));
                }
            }
            throw new RuntimeException("view must have a tag");
        }
        return null;
    }

    @Override // b.m.d
    public ViewDataBinding c(e eVar, View[] viewArr, int i) {
        if (viewArr == null || viewArr.length == 0 || f5672a.get(i) <= 0 || viewArr[0].getTag() != null) {
            return null;
        }
        throw new RuntimeException("view must have a tag");
    }
}