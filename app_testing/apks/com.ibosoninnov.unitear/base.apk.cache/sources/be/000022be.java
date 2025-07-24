package com.google.common.flogger.backend;

import c.b.a.a.a;
import com.google.common.flogger.util.Checks;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.Collections;
import java.util.Comparator;
import java.util.Map;
import java.util.SortedMap;
import java.util.SortedSet;
import java.util.TreeMap;
import java.util.TreeSet;

/* loaded from: classes.dex */
public final class Tags {
    private Integer hashCode;
    private final SortedMap<String, SortedSet<Object>> map;
    private String toString;
    private static final Comparator<Object> VALUE_COMPARATOR = new Comparator<Object>() { // from class: com.google.common.flogger.backend.Tags.1
        @Override // java.util.Comparator
        public int compare(Object obj, Object obj2) {
            Type of = Type.of(obj);
            Type of2 = Type.of(obj2);
            return of == of2 ? of.compare(obj, obj2) : of.compareTo(of2);
        }
    };
    private static final SortedSet<Object> EMPTY_SET = Collections.unmodifiableSortedSet(new TreeSet());
    private static final Tags EMPTY_TAGS = new Tags(Collections.unmodifiableSortedMap(new TreeMap()));

    /* loaded from: classes.dex */
    public enum Type {
        BOOLEAN { // from class: com.google.common.flogger.backend.Tags.Type.1
            @Override // com.google.common.flogger.backend.Tags.Type
            public int compare(Object obj, Object obj2) {
                return ((Boolean) obj).compareTo((Boolean) obj2);
            }
        },
        STRING { // from class: com.google.common.flogger.backend.Tags.Type.2
            @Override // com.google.common.flogger.backend.Tags.Type
            public int compare(Object obj, Object obj2) {
                return ((String) obj).compareTo((String) obj2);
            }
        },
        LONG { // from class: com.google.common.flogger.backend.Tags.Type.3
            @Override // com.google.common.flogger.backend.Tags.Type
            public int compare(Object obj, Object obj2) {
                return ((Long) obj).compareTo((Long) obj2);
            }
        },
        DOUBLE { // from class: com.google.common.flogger.backend.Tags.Type.4
            @Override // com.google.common.flogger.backend.Tags.Type
            public int compare(Object obj, Object obj2) {
                return ((Double) obj).compareTo((Double) obj2);
            }
        };

        /* JADX INFO: Access modifiers changed from: private */
        public static Type of(Object obj) {
            if (obj instanceof String) {
                return STRING;
            }
            if (obj instanceof Boolean) {
                return BOOLEAN;
            }
            if (obj instanceof Long) {
                return LONG;
            }
            if (obj instanceof Double) {
                return DOUBLE;
            }
            StringBuilder x = a.x("invalid tag type: ");
            x.append(obj.getClass());
            throw new AssertionError(x.toString());
        }

        public abstract int compare(Object obj, Object obj2);
    }

    public static Builder builder() {
        return new Builder();
    }

    public static Tags empty() {
        return EMPTY_TAGS;
    }

    public SortedMap<String, SortedSet<Object>> asMap() {
        return this.map;
    }

    public void emitAll(KeyValueHandler keyValueHandler) {
        for (Map.Entry<String, SortedSet<Object>> entry : this.map.entrySet()) {
            String key = entry.getKey();
            SortedSet<Object> value = entry.getValue();
            if (!value.isEmpty()) {
                for (Object obj : value) {
                    keyValueHandler.handle(key, obj);
                }
            } else {
                keyValueHandler.handle(key, null);
            }
        }
    }

    public boolean equals(Object obj) {
        return (obj instanceof Tags) && ((Tags) obj).map.equals(this.map);
    }

    public int hashCode() {
        if (this.hashCode == null) {
            this.hashCode = Integer.valueOf(this.map.hashCode());
        }
        return this.hashCode.intValue();
    }

    public boolean isEmpty() {
        return this.map.isEmpty();
    }

    public Tags merge(Tags tags) {
        if (tags.isEmpty()) {
            return this;
        }
        if (isEmpty()) {
            return tags;
        }
        TreeMap treeMap = new TreeMap();
        for (Map.Entry<String, SortedSet<Object>> entry : this.map.entrySet()) {
            SortedSet<Object> sortedSet = tags.map.get(entry.getKey());
            if (sortedSet != null && !entry.getValue().containsAll(sortedSet)) {
                if (sortedSet.containsAll(entry.getValue())) {
                    treeMap.put(entry.getKey(), sortedSet);
                } else {
                    TreeSet treeSet = new TreeSet((SortedSet) entry.getValue());
                    treeSet.addAll(sortedSet);
                    treeMap.put(entry.getKey(), Collections.unmodifiableSortedSet(treeSet));
                }
            } else {
                treeMap.put(entry.getKey(), entry.getValue());
            }
        }
        for (Map.Entry<String, SortedSet<Object>> entry2 : tags.map.entrySet()) {
            if (!this.map.containsKey(entry2.getKey())) {
                treeMap.put(entry2.getKey(), entry2.getValue());
            }
        }
        return new Tags(Collections.unmodifiableSortedMap(treeMap));
    }

    public String toString() {
        if (this.toString == null) {
            StringBuilder sb = new StringBuilder();
            KeyValueFormatter keyValueFormatter = new KeyValueFormatter("[ ", " ]", sb);
            emitAll(keyValueFormatter);
            keyValueFormatter.done();
            this.toString = sb.toString();
        }
        return this.toString;
    }

    /* loaded from: classes.dex */
    public static final class Builder {
        private final SortedMap<String, SortedSet<Object>> map = new TreeMap();

        private void addImpl(String str, Object obj) {
            SortedSet<Object> sortedSet = this.map.get(Checks.checkMetadataIdentifier(str));
            if (sortedSet == null || sortedSet == Tags.EMPTY_SET) {
                sortedSet = new TreeSet<>(Tags.VALUE_COMPARATOR);
                this.map.put(str, sortedSet);
            }
            sortedSet.add(obj);
        }

        @CanIgnoreReturnValue
        public Builder addTag(String str) {
            if (this.map.get(Checks.checkMetadataIdentifier(str)) == null) {
                this.map.put(str, Tags.EMPTY_SET);
            }
            return this;
        }

        public Tags build() {
            if (this.map.isEmpty()) {
                return Tags.EMPTY_TAGS;
            }
            TreeMap treeMap = new TreeMap();
            for (Map.Entry<String, SortedSet<Object>> entry : this.map.entrySet()) {
                treeMap.put(entry.getKey(), Collections.unmodifiableSortedSet(new TreeSet((SortedSet) entry.getValue())));
            }
            return new Tags(Collections.unmodifiableSortedMap(treeMap));
        }

        public String toString() {
            return build().toString();
        }

        @CanIgnoreReturnValue
        public Builder addTag(String str, String str2) {
            if (str2 != null) {
                addImpl(str, str2);
                return this;
            }
            throw new IllegalArgumentException("tag values cannot be null");
        }

        @CanIgnoreReturnValue
        public Builder addTag(String str, boolean z) {
            addImpl(str, Boolean.valueOf(z));
            return this;
        }

        @CanIgnoreReturnValue
        public Builder addTag(String str, long j) {
            addImpl(str, Long.valueOf(j));
            return this;
        }

        @CanIgnoreReturnValue
        public Builder addTag(String str, double d2) {
            addImpl(str, Double.valueOf(d2));
            return this;
        }
    }

    private Tags(SortedMap<String, SortedSet<Object>> sortedMap) {
        this.hashCode = null;
        this.toString = null;
        this.map = sortedMap;
    }
}