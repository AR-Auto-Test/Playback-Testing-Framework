package com.google.protobuf;

import c.b.a.a.a;
import com.google.protobuf.Descriptors;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.logging.Logger;

/* loaded from: classes2.dex */
public class TypeRegistry {
    private static final Logger logger = Logger.getLogger(TypeRegistry.class.getName());
    private final Map<String, Descriptors.Descriptor> types;

    /* loaded from: classes2.dex */
    public static final class Builder {
        private final Set<String> files;
        private Map<String, Descriptors.Descriptor> types;

        private void addFile(Descriptors.FileDescriptor fileDescriptor) {
            if (this.files.add(fileDescriptor.getFullName())) {
                for (Descriptors.FileDescriptor fileDescriptor2 : fileDescriptor.getDependencies()) {
                    addFile(fileDescriptor2);
                }
                for (Descriptors.Descriptor descriptor : fileDescriptor.getMessageTypes()) {
                    addMessage(descriptor);
                }
            }
        }

        private void addMessage(Descriptors.Descriptor descriptor) {
            for (Descriptors.Descriptor descriptor2 : descriptor.getNestedTypes()) {
                addMessage(descriptor2);
            }
            if (this.types.containsKey(descriptor.getFullName())) {
                Logger logger = TypeRegistry.logger;
                StringBuilder x = a.x("Type ");
                x.append(descriptor.getFullName());
                x.append(" is added multiple times.");
                logger.warning(x.toString());
                return;
            }
            this.types.put(descriptor.getFullName(), descriptor);
        }

        public Builder add(Descriptors.Descriptor descriptor) {
            if (this.types != null) {
                addFile(descriptor.getFile());
                return this;
            }
            throw new IllegalStateException("A TypeRegistry.Builder can only be used once.");
        }

        public TypeRegistry build() {
            TypeRegistry typeRegistry = new TypeRegistry(this.types);
            this.types = null;
            return typeRegistry;
        }

        private Builder() {
            this.files = new HashSet();
            this.types = new HashMap();
        }

        public Builder add(Iterable<Descriptors.Descriptor> iterable) {
            if (this.types != null) {
                for (Descriptors.Descriptor descriptor : iterable) {
                    addFile(descriptor.getFile());
                }
                return this;
            }
            throw new IllegalStateException("A TypeRegistry.Builder can only be used once.");
        }
    }

    /* loaded from: classes2.dex */
    public static class EmptyTypeRegistryHolder {
        private static final TypeRegistry EMPTY = new TypeRegistry(Collections.emptyMap());

        private EmptyTypeRegistryHolder() {
        }
    }

    public TypeRegistry(Map<String, Descriptors.Descriptor> map) {
        this.types = map;
    }

    public static TypeRegistry getEmptyTypeRegistry() {
        return EmptyTypeRegistryHolder.EMPTY;
    }

    private static String getTypeName(String str) {
        String[] split = str.split("/");
        if (split.length != 1) {
            return split[split.length - 1];
        }
        throw new InvalidProtocolBufferException(a.q("Invalid type url found: ", str));
    }

    public static Builder newBuilder() {
        return new Builder();
    }

    public Descriptors.Descriptor find(String str) {
        return this.types.get(str);
    }

    public final Descriptors.Descriptor getDescriptorForTypeUrl(String str) {
        return find(getTypeName(str));
    }
}