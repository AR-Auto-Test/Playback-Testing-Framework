package b.w.b;

import android.os.Bundle;
import android.view.View;
import android.view.ViewGroup;
import android.view.accessibility.AccessibilityEvent;
import androidx.recyclerview.widget.RecyclerView;
import java.util.Map;
import java.util.WeakHashMap;

/* compiled from: RecyclerViewAccessibilityDelegate.java */
/* loaded from: classes.dex */
public class v extends b.j.j.a {
    private final a mItemDelegate;
    public final RecyclerView mRecyclerView;

    /* compiled from: RecyclerViewAccessibilityDelegate.java */
    /* loaded from: classes.dex */
    public static class a extends b.j.j.a {

        /* renamed from: a  reason: collision with root package name */
        public final v f2800a;

        /* renamed from: b  reason: collision with root package name */
        public Map<View, b.j.j.a> f2801b = new WeakHashMap();

        public a(v vVar) {
            this.f2800a = vVar;
        }

        @Override // b.j.j.a
        public boolean dispatchPopulateAccessibilityEvent(View view, AccessibilityEvent accessibilityEvent) {
            b.j.j.a aVar = this.f2801b.get(view);
            if (aVar != null) {
                return aVar.dispatchPopulateAccessibilityEvent(view, accessibilityEvent);
            }
            return super.dispatchPopulateAccessibilityEvent(view, accessibilityEvent);
        }

        @Override // b.j.j.a
        public b.j.j.x.c getAccessibilityNodeProvider(View view) {
            b.j.j.a aVar = this.f2801b.get(view);
            if (aVar != null) {
                return aVar.getAccessibilityNodeProvider(view);
            }
            return super.getAccessibilityNodeProvider(view);
        }

        @Override // b.j.j.a
        public void onInitializeAccessibilityEvent(View view, AccessibilityEvent accessibilityEvent) {
            b.j.j.a aVar = this.f2801b.get(view);
            if (aVar != null) {
                aVar.onInitializeAccessibilityEvent(view, accessibilityEvent);
            } else {
                super.onInitializeAccessibilityEvent(view, accessibilityEvent);
            }
        }

        @Override // b.j.j.a
        public void onInitializeAccessibilityNodeInfo(View view, b.j.j.x.b bVar) {
            if (!this.f2800a.shouldIgnore() && this.f2800a.mRecyclerView.getLayoutManager() != null) {
                this.f2800a.mRecyclerView.getLayoutManager().onInitializeAccessibilityNodeInfoForItem(view, bVar);
                b.j.j.a aVar = this.f2801b.get(view);
                if (aVar != null) {
                    aVar.onInitializeAccessibilityNodeInfo(view, bVar);
                    return;
                } else {
                    super.onInitializeAccessibilityNodeInfo(view, bVar);
                    return;
                }
            }
            super.onInitializeAccessibilityNodeInfo(view, bVar);
        }

        @Override // b.j.j.a
        public void onPopulateAccessibilityEvent(View view, AccessibilityEvent accessibilityEvent) {
            b.j.j.a aVar = this.f2801b.get(view);
            if (aVar != null) {
                aVar.onPopulateAccessibilityEvent(view, accessibilityEvent);
            } else {
                super.onPopulateAccessibilityEvent(view, accessibilityEvent);
            }
        }

        @Override // b.j.j.a
        public boolean onRequestSendAccessibilityEvent(ViewGroup viewGroup, View view, AccessibilityEvent accessibilityEvent) {
            b.j.j.a aVar = this.f2801b.get(viewGroup);
            if (aVar != null) {
                return aVar.onRequestSendAccessibilityEvent(viewGroup, view, accessibilityEvent);
            }
            return super.onRequestSendAccessibilityEvent(viewGroup, view, accessibilityEvent);
        }

        @Override // b.j.j.a
        public boolean performAccessibilityAction(View view, int i, Bundle bundle) {
            if (!this.f2800a.shouldIgnore() && this.f2800a.mRecyclerView.getLayoutManager() != null) {
                b.j.j.a aVar = this.f2801b.get(view);
                if (aVar != null) {
                    if (aVar.performAccessibilityAction(view, i, bundle)) {
                        return true;
                    }
                } else if (super.performAccessibilityAction(view, i, bundle)) {
                    return true;
                }
                return this.f2800a.mRecyclerView.getLayoutManager().performAccessibilityActionForItem(view, i, bundle);
            }
            return super.performAccessibilityAction(view, i, bundle);
        }

        @Override // b.j.j.a
        public void sendAccessibilityEvent(View view, int i) {
            b.j.j.a aVar = this.f2801b.get(view);
            if (aVar != null) {
                aVar.sendAccessibilityEvent(view, i);
            } else {
                super.sendAccessibilityEvent(view, i);
            }
        }

        @Override // b.j.j.a
        public void sendAccessibilityEventUnchecked(View view, AccessibilityEvent accessibilityEvent) {
            b.j.j.a aVar = this.f2801b.get(view);
            if (aVar != null) {
                aVar.sendAccessibilityEventUnchecked(view, accessibilityEvent);
            } else {
                super.sendAccessibilityEventUnchecked(view, accessibilityEvent);
            }
        }
    }

    public v(RecyclerView recyclerView) {
        this.mRecyclerView = recyclerView;
        b.j.j.a itemDelegate = getItemDelegate();
        if (itemDelegate != null && (itemDelegate instanceof a)) {
            this.mItemDelegate = (a) itemDelegate;
        } else {
            this.mItemDelegate = new a(this);
        }
    }

    public b.j.j.a getItemDelegate() {
        return this.mItemDelegate;
    }

    @Override // b.j.j.a
    public void onInitializeAccessibilityEvent(View view, AccessibilityEvent accessibilityEvent) {
        super.onInitializeAccessibilityEvent(view, accessibilityEvent);
        if (!(view instanceof RecyclerView) || shouldIgnore()) {
            return;
        }
        RecyclerView recyclerView = (RecyclerView) view;
        if (recyclerView.getLayoutManager() != null) {
            recyclerView.getLayoutManager().onInitializeAccessibilityEvent(accessibilityEvent);
        }
    }

    @Override // b.j.j.a
    public void onInitializeAccessibilityNodeInfo(View view, b.j.j.x.b bVar) {
        super.onInitializeAccessibilityNodeInfo(view, bVar);
        if (shouldIgnore() || this.mRecyclerView.getLayoutManager() == null) {
            return;
        }
        this.mRecyclerView.getLayoutManager().onInitializeAccessibilityNodeInfo(bVar);
    }

    @Override // b.j.j.a
    public boolean performAccessibilityAction(View view, int i, Bundle bundle) {
        if (super.performAccessibilityAction(view, i, bundle)) {
            return true;
        }
        if (shouldIgnore() || this.mRecyclerView.getLayoutManager() == null) {
            return false;
        }
        return this.mRecyclerView.getLayoutManager().performAccessibilityAction(i, bundle);
    }

    public boolean shouldIgnore() {
        return this.mRecyclerView.hasPendingAdapterUpdates();
    }
}